"""Operator-facing visualization for four-orange grounding."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tiptop.orange_demo import GroundingResult


_SELECTED_COLOR = (34, 197, 94, 255)
_ORANGE_COLOR = (249, 115, 22, 255)
_CONTAINER_COLOR = (14, 165, 233, 255)
_OTHER_COLOR = (148, 163, 184, 255)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _pixel_box(bbox: dict, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    pixels = bbox.get("box_2d_pixels")
    if isinstance(pixels, (list, tuple)) and len(pixels) == 4:
        x1, y1, x2, y2 = (int(round(float(value))) for value in pixels)
    else:
        coords = bbox.get("box_2d")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            return None
        ymin, xmin, ymax, xmax = (float(value) for value in coords)
        x1 = int(round(xmin / 1000.0 * image_width))
        y1 = int(round(ymin / 1000.0 * image_height))
        x2 = int(round(xmax / 1000.0 * image_width))
        y2 = int(round(ymax / 1000.0 * image_height))

    x1 = min(max(x1, 0), image_width - 1)
    y1 = min(max(y1, 0), image_height - 1)
    x2 = min(max(x2, 0), image_width - 1)
    y2 = min(max(y2, 0), image_height - 1)
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def _goal_text(grounded_atoms: list[dict]) -> str:
    goals = []
    for atom in grounded_atoms:
        predicate = atom.get("predicate", "?")
        args = ", ".join(str(arg) for arg in atom.get("args", []))
        goals.append(f"{predicate}({args})")
    return "; ".join(goals) if goals else "none"


def orange_grounding_metadata(
    task_instruction: str,
    result: GroundingResult,
    grounded_atoms: list[dict] | None = None,
) -> dict:
    """Build the JSON-serializable grounding record saved with each run."""
    atoms = result.grounded_atoms if grounded_atoms is None else grounded_atoms
    return {
        "instruction": task_instruction,
        "selected_label": result.selected_label,
        "selector": asdict(result.selector) if result.selector is not None else None,
        "grounded_atoms": atoms,
        "bboxes": result.bboxes,
    }


def render_orange_grounding_overlay(
    rgb: np.ndarray,
    result: GroundingResult,
    grounded_atoms: list[dict] | None = None,
) -> np.ndarray:
    """Render ranked orange boxes, the selected target, and the symbolic goal."""
    image_array = np.asarray(rgb)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Expected RGB image shaped (H, W, 3), got {image_array.shape}.")

    image = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, mode="RGBA")
    label_font = _font(max(16, round(image.width / 80)))
    banner_font = _font(max(18, round(image.width / 64)))

    for bbox in result.bboxes:
        label = str(bbox.get("label", "unknown"))
        coords = _pixel_box(bbox, image.width, image.height)
        if coords is None:
            continue
        x1, y1, x2, y2 = coords
        is_selected = label == result.selected_label
        is_orange = label.startswith("orange_")
        is_container = any(
            term in label.lower()
            for term in ("box", "bin", "container", "basket", "carton", "crate")
        )
        color = (
            _SELECTED_COLOR
            if is_selected
            else _ORANGE_COLOR
            if is_orange
            else _CONTAINER_COLOR
            if is_container
            else _OTHER_COLOR
        )
        width = 7 if is_selected else 3
        if is_selected:
            draw.rectangle(coords, fill=(*color[:3], 45))
        draw.rectangle(coords, outline=color, width=width)

        label_text = f"TARGET · {label}" if is_selected else label
        text_bbox = draw.textbbox((x1, y1), label_text, font=label_font)
        text_height = text_bbox[3] - text_bbox[1]
        text_width = text_bbox[2] - text_bbox[0]
        tag_y = max(0, y1 - text_height - 8)
        draw.rounded_rectangle(
            (x1, tag_y, min(image.width - 1, x1 + text_width + 12), tag_y + text_height + 8),
            radius=4,
            fill=color,
        )
        draw.text((x1 + 6, tag_y + 3), label_text, fill=(255, 255, 255, 255), font=label_font)

    selector = result.selector
    if selector is None:
        selector_text = "selector: none"
    else:
        direction = "right-to-left" if selector.reverse else "left-to-right"
        if selector.axis == "y":
            direction = "bottom-to-top" if selector.reverse else "top-to-bottom"
        selector_text = f"selector: {selector.axis}-axis rank {selector.rank} ({direction})"
    atoms = result.grounded_atoms if grounded_atoms is None else grounded_atoms
    selected_text = result.selected_label or "none"
    lines = [
        "FOUR-ORANGE GROUNDING",
        f"selected: {selected_text}  |  {selector_text}  |  goal: {_goal_text(atoms)}",
    ]
    line_height = banner_font.size + 8 if hasattr(banner_font, "size") else 28
    banner_height = line_height * len(lines) + 10
    draw.rectangle((0, 0, image.width, banner_height), fill=(15, 23, 42, 220))
    for index, line in enumerate(lines):
        draw.text((14, 5 + index * line_height), line, fill=(255, 255, 255, 255), font=banner_font)

    return np.asarray(image)


def save_orange_grounding_visualization(
    save_dir: Path,
    overlay: np.ndarray,
    metadata: dict,
) -> tuple[Path, Path]:
    """Save the quick-look overlay and machine-readable grounding record."""
    save_dir.mkdir(parents=True, exist_ok=True)
    perception_dir = save_dir / "perception"
    perception_dir.mkdir(exist_ok=True)
    overlay_path = save_dir / "orange_selection_viz.png"
    metadata_path = perception_dir / "orange_grounding.json"
    Image.fromarray(overlay).save(overlay_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return overlay_path, metadata_path


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
