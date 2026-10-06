"""Shared bounding-box coordinate conversions."""

import logging

import numpy as np

_log = logging.getLogger(__name__)


def _to_norm_1000(value: float, dimension: int) -> int:
    return int(round(value / dimension * 1000))


def _pixel_xyxy_from_gemini_norm(box_2d: list, img_width: int, img_height: int) -> tuple[int, int, int, int] | None:
    if len(box_2d) != 4:
        return None
    ymin, xmin, ymax, xmax = box_2d
    if ymax <= ymin or xmax <= xmin:
        return None
    return (
        int((xmin / 1000.0) * img_width),
        int((ymin / 1000.0) * img_height),
        int((xmax / 1000.0) * img_width),
        int((ymax / 1000.0) * img_height),
    )


def _gemini_norm_from_pixel_xyxy(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int) -> list:
    return [
        _to_norm_1000(y1, img_height),
        _to_norm_1000(x1, img_width),
        _to_norm_1000(y2, img_height),
        _to_norm_1000(x2, img_width),
    ]


def _valid_pixel_xyxy(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int) -> bool:
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < -2 or y1 < -2 or x2 > img_width + 2 or y2 > img_height + 2:
        return False
    area = (x2 - x1) * (y2 - y1)
    img_area = img_width * img_height
    return 0.00005 * img_area <= area <= 0.85 * img_area


def _valid_gemini_norm(box_2d: list) -> bool:
    if len(box_2d) != 4:
        return False
    ymin, xmin, ymax, xmax = box_2d
    if ymax <= ymin or xmax <= xmin:
        return False
    if any(v < 0 or v > 1000 for v in box_2d):
        return False
    return (ymax - ymin) * (xmax - xmin) <= 850_000


def _all_coords_in_0_1000(raw_box: list) -> bool:
    return len(raw_box) == 4 and all(0 <= int(round(v)) <= 1000 for v in raw_box)


def _coords_ambiguous_between_pixel_and_gemini(raw_boxes: list[list], img_width: int, img_height: int) -> bool:
    """True when values fit 0-1000 but the image is HD, so pixel vs gemini-norm is ambiguous."""
    if max(img_width, img_height) <= 1000:
        return False
    if not raw_boxes:
        return False
    if any(int(round(v)) > 1000 for raw in raw_boxes for v in raw):
        return False
    return all(_all_coords_in_0_1000(raw) for raw in raw_boxes)


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, (box[2] - box[0]) * (box[3] - box[1]))


def _prefer_gemini_over_pixel(raw_box: list, img_width: int, img_height: int) -> bool:
    """Prefer 0-1000 [ymin,xmin,ymax,xmax] when pixel xyxy would shift the box too far right."""
    if not _coords_ambiguous_between_pixel_and_gemini([raw_box], img_width, img_height):
        return False
    a, b, c, d = [int(round(v)) for v in raw_box]
    px = (a, b, c, d)
    gpx = _pixel_xyxy_from_gemini_norm([a, b, c, d], img_width, img_height)
    if gpx is None or not _valid_pixel_xyxy(*px, img_width, img_height):
        return False
    # GPT-4o often returns gemini-norm coords; pixel xyxy on HD images misplaces left-side objects.
    return px[0] > 0.25 * img_width and gpx[0] < px[0] - 50


def _prefer_gemini_joint_layout(raw_boxes: list[list], img_width: int, img_height: int) -> bool:
    """Jointly prefer gemini-norm when the largest pixel box is shifted right vs gemini."""
    if not _coords_ambiguous_between_pixel_and_gemini(raw_boxes, img_width, img_height):
        return False
    pixel_joint: list[tuple[int, int, int, int]] = []
    gemini_joint: list[tuple[int, int, int, int]] = []
    for raw in raw_boxes:
        a, b, c, d = [int(round(v)) for v in raw]
        pixel_joint.append((a, b, c, d))
        gpx = _pixel_xyxy_from_gemini_norm([a, b, c, d], img_width, img_height)
        if gpx is None:
            return False
        gemini_joint.append(gpx)
    pi = max(range(len(pixel_joint)), key=lambda i: _box_area(pixel_joint[i]))
    gi = max(range(len(gemini_joint)), key=lambda i: _box_area(gemini_joint[i]))
    return pixel_joint[pi][0] > 0.25 * img_width and gemini_joint[gi][0] < pixel_joint[pi][0] - 50


def _score_pixel_layout(pixel_boxes: list[tuple[int, int, int, int]], img_width: int, img_height: int) -> float:
    if not pixel_boxes:
        return -1e9
    img_area = img_width * img_height
    score = 0.0
    areas = []
    for x1, y1, x2, y2 in pixel_boxes:
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)
        frac = area / img_area
        if frac < 0.0002:
            score -= 5.0
        elif frac > 0.7:
            score -= 3.0
        else:
            score += 1.0
    if areas:
        # Prefer a clear largest box (container) when multiple objects are present.
        max_frac = max(areas) / img_area
        if len(areas) >= 2 and max_frac > 0.03:
            score += 2.0
    return score


def apply_gemini_norm_bboxes(bboxes: list[dict], img_width: int, img_height: int) -> None:
    """In-place: box_2d is already [ymin, xmin, ymax, xmax] on 0-1000 grid (no format guessing)."""
    for bbox in bboxes:
        raw = bbox.get("box_2d", [])
        if len(raw) != 4:
            continue
        ymin, xmin, ymax, xmax = [int(round(v)) for v in raw]
        gemini = [ymin, xmin, ymax, xmax]
        pixels = _pixel_xyxy_from_gemini_norm(gemini, img_width, img_height)
        bbox["box_2d_raw"] = gemini
        bbox["box_2d"] = gemini
        bbox["box_2d_pixels"] = list(pixels) if pixels else []
        bbox["box_2d_format"] = "gemini_norm"


def parse_openai_box_2d(raw_box: list, img_width: int, img_height: int) -> tuple[list, list[int], str]:
    """Parse raw OpenAI box into internal gemini-norm + pixel xyxy.

    OpenAI may return either:
      - pixel [xmin, ymin, xmax, ymax]
      - normalized gemini [ymin, xmin, ymax, xmax] in 0-1000
    """
    if len(raw_box) != 4:
        return raw_box, [], "invalid"

    a, b, c, d = [int(round(v)) for v in raw_box]
    candidates: list[tuple[str, list, list[int], float]] = []

    # Pixel [xmin, ymin, xmax, ymax]
    px = (a, b, c, d)
    if _valid_pixel_xyxy(*px, img_width, img_height):
        gemini = _gemini_norm_from_pixel_xyxy(*px, img_width, img_height)
        candidates.append(("pixel_xyxy", gemini, list(px), _score_pixel_layout([px], img_width, img_height)))

    # Gemini normalized [ymin, xmin, ymax, xmax]
    if _valid_gemini_norm([a, b, c, d]):
        px2 = _pixel_xyxy_from_gemini_norm([a, b, c, d], img_width, img_height)
        if px2 and _valid_pixel_xyxy(*px2, img_width, img_height):
            candidates.append(
                ("gemini_norm", [a, b, c, d], list(px2), _score_pixel_layout([px2], img_width, img_height))
            )

    # Swapped normalized [xmin, ymin, xmax, ymax] in 0-1000
    if _valid_gemini_norm([b, a, d, c]):
        gemini = [b, a, d, c]
        px3 = _pixel_xyxy_from_gemini_norm(gemini, img_width, img_height)
        if px3 and _valid_pixel_xyxy(*px3, img_width, img_height):
            candidates.append(
                ("norm_xyxy", gemini, list(px3), _score_pixel_layout([px3], img_width, img_height))
            )

    if not candidates:
        _log.warning(f"Could not parse OpenAI box {raw_box}; passing through unchanged")
        return raw_box, [], "unknown"

    def _rank(item: tuple[str, list, list[int], float]) -> tuple[float, int]:
        fmt, _, _, score = item
        prefer_gemini = 1 if fmt == "gemini_norm" and _prefer_gemini_over_pixel(raw_box, img_width, img_height) else 0
        return (score, prefer_gemini)

    fmt, gemini, pixels, _ = max(candidates, key=_rank)
    return gemini, pixels, fmt


def normalize_openai_bboxes(bboxes: list[dict], img_width: int, img_height: int) -> None:
    """In-place parse OpenAI boxes into internal gemini-norm + pixel xyxy."""
    parsed_pixels: list[tuple[int, int, int, int]] = []
    parsed_entries: list[tuple[dict, list, list[int], str]] = []

    for bbox in bboxes:
        raw = bbox.get("box_2d", [])
        gemini, pixels, fmt = parse_openai_box_2d(raw, img_width, img_height)
        parsed_entries.append((bbox, gemini, pixels, fmt))
        if pixels:
            parsed_pixels.append(tuple(pixels))

    # Global preference: if every box can be pixel xyxy, prefer that interpretation jointly.
    all_raw = [bbox.get("box_2d", []) for bbox in bboxes if len(bbox.get("box_2d", [])) == 4]
    pixel_joint = []
    pixel_valid = True
    for raw in all_raw:
        a, b, c, d = [int(round(v)) for v in raw]
        px = (a, b, c, d)
        if not _valid_pixel_xyxy(*px, img_width, img_height):
            pixel_valid = False
            break
        pixel_joint.append(px)

    prefer_gemini_joint = _prefer_gemini_joint_layout(all_raw, img_width, img_height)
    use_joint_pixels = (
        pixel_valid
        and not prefer_gemini_joint
        and _score_pixel_layout(pixel_joint, img_width, img_height)
        >= _score_pixel_layout(parsed_pixels, img_width, img_height) - 0.5
    )

    for bbox, gemini, pixels, fmt in parsed_entries:
        raw = bbox.get("box_2d", [])
        bbox["box_2d_raw"] = raw
        if prefer_gemini_joint and len(raw) == 4:
            a, b, c, d = [int(round(v)) for v in raw]
            gemini = [a, b, c, d]
            gpx = _pixel_xyxy_from_gemini_norm(gemini, img_width, img_height)
            pixels = list(gpx) if gpx else pixels
            fmt = "gemini_norm"
        elif use_joint_pixels and len(raw) == 4:
            a, b, c, d = [int(round(v)) for v in raw]
            pixels = [a, b, c, d]
            gemini = _gemini_norm_from_pixel_xyxy(a, b, c, d, img_width, img_height)
            fmt = "pixel_xyxy"
        bbox["box_2d_pixels"] = pixels
        bbox["box_2d"] = gemini
        bbox["box_2d_format"] = fmt


def refine_bboxes_from_masks(bboxes: list[dict], masks: np.ndarray, img_width: int, img_height: int) -> None:
    """Tighten bbox labels using SAM mask extents (more accurate than VLM boxes)."""
    if masks is None or len(masks) == 0:
        return
    masks_sv = masks.squeeze(1) if masks.ndim == 4 else masks
    for i, bbox in enumerate(bboxes):
        if i >= len(masks_sv):
            break
        mask = masks_sv[i].astype(bool)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        if not _valid_pixel_xyxy(x1, y1, x2, y2, img_width, img_height):
            continue
        bbox["box_2d_pixels"] = [x1, y1, x2, y2]
        bbox["box_2d"] = _gemini_norm_from_pixel_xyxy(x1, y1, x2, y2, img_width, img_height)
        bbox["box_2d_format"] = "sam_mask"


def box_2d_to_pixel_xyxy(box_2d: list, img_width: int, img_height: int) -> tuple[int, int, int, int] | None:
    """Convert normalized [ymin, xmin, ymax, xmax] (0-1000) to pixel [x1, y1, x2, y2]."""
    if len(box_2d) != 4:
        return None
    return _pixel_xyxy_from_gemini_norm(box_2d, img_width, img_height)


def bbox_to_pixel_xyxy(bbox: dict, img_width: int, img_height: int) -> tuple[int, int, int, int] | None:
    """Prefer stored pixel xyxy; fall back to normalized conversion."""
    pixels = bbox.get("box_2d_pixels")
    if pixels and len(pixels) == 4:
        x1, y1, x2, y2 = [int(v) for v in pixels]
        if _valid_pixel_xyxy(x1, y1, x2, y2, img_width, img_height):
            return x1, y1, x2, y2
    return box_2d_to_pixel_xyxy(bbox.get("box_2d", []), img_width, img_height)


def boxes_2d_to_pixel_xyxy(bboxes: list[dict], img_width: int, img_height: int) -> np.ndarray:
    """Convert detection dicts to (N, 4) pixel boxes [x1, y1, x2, y2]."""
    xyxy = []
    for bbox in bboxes:
        coords = bbox_to_pixel_xyxy(bbox, img_width, img_height)
        if coords is None:
            continue
        xyxy.append(coords)
    if not xyxy:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(xyxy, dtype=np.float32)


def boxes_2d_to_sam_xyxy(bboxes: list[dict], img_width: int, img_height: int) -> np.ndarray:
    """Convert detection dicts to SAM2 format [x0, y0, x1, y1] in pixels."""
    return boxes_2d_to_pixel_xyxy(bboxes, img_width, img_height)


def boxes_containment_score(bboxes: list[dict], container_label: str, inner_labels: list[str]) -> float:
    """Fraction of inner boxes whose center lies inside the container box."""
    by_label = {b["label"]: b for b in bboxes}
    container = by_label.get(container_label)
    if container is None:
        return 1.0
    cp = container.get("box_2d_pixels")
    if not cp:
        return 1.0
    cx1, cy1, cx2, cy2 = cp
    hits = 0
    total = 0
    for label in inner_labels:
        inner = by_label.get(label)
        if inner is None:
            continue
        ip = inner.get("box_2d_pixels")
        if not ip:
            continue
        ix1, iy1, ix2, iy2 = ip
        icx = (ix1 + ix2) / 2
        icy = (iy1 + iy2) / 2
        total += 1
        if cx1 <= icx <= cx2 and cy1 <= icy <= cy2:
            hits += 1
    if total == 0:
        return 1.0
    return hits / total
