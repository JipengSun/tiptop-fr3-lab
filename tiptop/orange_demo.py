"""Deterministic language grounding for the four-orange tabletop demo."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal


class OrangeGroundingError(ValueError):
    """Raised when the orange target cannot be grounded safely."""


@dataclass(frozen=True)
class OrangeSelector:
    """A visual rank such as "second from left" or "topmost"."""

    axis: Literal["x", "y"]
    rank: int
    reverse: bool
    description: str


@dataclass(frozen=True)
class GroundingResult:
    """Post-processed TiPToP detections and symbolic goals."""

    bboxes: list[dict]
    grounded_atoms: list[dict]
    selected_label: str | None
    selector: OrangeSelector | None


_ORDINALS = {
    "first": 1,
    "1st": 1,
    "one": 1,
    "second": 2,
    "2nd": 2,
    "two": 2,
    "third": 3,
    "3rd": 3,
    "three": 3,
    "fourth": 4,
    "4th": 4,
    "four": 4,
}
_ZH_NUMBERS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "1": 1, "2": 2, "3": 3, "4": 4}
_CONTAINER_TERMS = ("box", "bin", "container", "basket", "carton", "crate", "盒", "筐", "箱")


def _selector(axis: Literal["x", "y"], rank: int, reverse: bool, description: str) -> OrangeSelector:
    return OrangeSelector(axis=axis, rank=rank, reverse=reverse, description=description)


def parse_orange_selector(instruction: str) -> OrangeSelector | None:
    """Parse a supported visual or ordinal reference to one orange."""
    text = " ".join(instruction.lower().replace("-", " ").split())

    relative_patterns = (
        (
            r"\b(first|1st|second|2nd|third|3rd|fourth|4th)\s+(?:orange\s+)?from\s+the\s+left\b",
            "x",
            False,
        ),
        (
            r"\b(first|1st|second|2nd|third|3rd|fourth|4th)\s+(?:orange\s+)?from\s+the\s+right\b",
            "x",
            True,
        ),
        (r"\bleft\s+(first|1st|second|2nd|third|3rd|fourth|4th)\s+orange\b", "x", False),
        (r"\bright\s+(first|1st|second|2nd|third|3rd|fourth|4th)\s+orange\b", "x", True),
    )
    for pattern, axis, reverse in relative_patterns:
        if match := re.search(pattern, text):
            token = match.group(1)
            return _selector(axis, _ORDINALS[token], reverse, match.group(0))

    named_extremes = (
        (r"\b(?:the\s+)?(?:leftmost|far left|left)\s+orange\b", "x", False, "leftmost orange"),
        (r"\b(?:the\s+)?(?:rightmost|far right|right)\s+orange\b", "x", True, "rightmost orange"),
        (r"\b(?:the\s+)?(?:topmost|uppermost|top)\s+orange\b", "y", False, "topmost orange"),
        (r"\b(?:the\s+)?(?:bottommost|lowermost|bottom)\s+orange\b", "y", True, "bottommost orange"),
    )
    for pattern, axis, reverse, description in named_extremes:
        if re.search(pattern, text):
            return _selector(axis, 1, reverse, description)

    ordinal_patterns = (
        r"\b(first|1st|second|2nd|third|3rd|fourth|4th)\s+orange\b",
        r"\borange\s+(?:number\s*)?#?\s*([1-4])\b",
        r"\borange_?([1-4])\b",
    )
    for pattern in ordinal_patterns:
        if match := re.search(pattern, text):
            token = match.group(1)
            rank = _ORDINALS.get(token, int(token) if token.isdigit() else 0)
            return _selector("x", rank, False, f"orange {rank} from the left")

    chinese_extremes = (
        (r"(?:最左(?:边)?|左边(?:的)?)橙子", "x", False, "最左边的橙子"),
        (r"(?:最右(?:边)?|右边(?:的)?)橙子", "x", True, "最右边的橙子"),
        (r"(?:最上(?:边)?|上边(?:的)?)橙子", "y", False, "最上边的橙子"),
        (r"(?:最下(?:边)?|下边(?:的)?)橙子", "y", True, "最下边的橙子"),
    )
    for pattern, axis, reverse, description in chinese_extremes:
        if re.search(pattern, instruction):
            return _selector(axis, 1, reverse, description)

    chinese_relative_patterns = (
        (r"从左(?:边)?(?:数)?第([一二两三四1-4])个?橙子", "x", False),
        (r"左(?:边)?第([一二两三四1-4])个?橙子", "x", False),
        (r"从右(?:边)?(?:数)?第([一二两三四1-4])个?橙子", "x", True),
        (r"右(?:边)?第([一二两三四1-4])个?橙子", "x", True),
    )
    for pattern, axis, reverse in chinese_relative_patterns:
        if match := re.search(pattern, instruction):
            rank = _ZH_NUMBERS[match.group(1)]
            return _selector(axis, rank, reverse, match.group(0))

    if match := re.search(r"第([一二两三四1-4])个?橙子", instruction):
        rank = _ZH_NUMBERS[match.group(1)]
        return _selector("x", rank, False, f"从左数第{rank}个橙子")
    if match := re.search(r"([一二两三四1-4])号橙子", instruction):
        rank = _ZH_NUMBERS[match.group(1)]
        return _selector("x", rank, False, f"{rank}号橙子（从左数）")

    if re.fullmatch(r"[1-4]", text):
        return _selector("x", int(text), False, f"orange {text} from the left")
    return None


def _normalized_label(label: str) -> str:
    return re.sub(r"[^a-z0-9_\u4e00-\u9fff]+", "_", label.lower()).strip("_")


def _is_container(label: str) -> bool:
    normalized = _normalized_label(label)
    return any(term in normalized for term in _CONTAINER_TERMS)


def _is_orange(label: str) -> bool:
    normalized = _normalized_label(label)
    return not _is_container(normalized) and (
        re.search(r"(?:^|_)orange(?:_|[0-9]|$)", normalized) is not None
        or "tangerine" in normalized
        or "mandarin" in normalized
        or "橙子" in normalized
    )


def _instruction_mentions_container(instruction: str) -> bool:
    return any(term in _normalized_label(instruction) for term in _CONTAINER_TERMS)


def _bbox_center(bbox: dict) -> tuple[float, float]:
    coords = bbox.get("box_2d")
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        raise OrangeGroundingError(f"Orange detection '{bbox.get('label', '<unnamed>')}' has an invalid box_2d.")
    try:
        ymin, xmin, ymax, xmax = (float(value) for value in coords)
    except (TypeError, ValueError) as exc:
        raise OrangeGroundingError(
            f"Orange detection '{bbox.get('label', '<unnamed>')}' has non-numeric box coordinates."
        ) from exc
    if xmin >= xmax or ymin >= ymax:
        raise OrangeGroundingError(
            f"Orange detection '{bbox.get('label', '<unnamed>')}' has a degenerate bounding box."
        )
    return (xmin + xmax) / 2, (ymin + ymax) / 2


def _destination_label(bboxes: list[dict], grounded_atoms: list[dict], orange_labels: set[str]) -> str:
    destinations = {
        atom["args"][1]
        for atom in grounded_atoms
        if atom.get("predicate") == "on"
        and len(atom.get("args", [])) == 2
        and atom["args"][0] in orange_labels
        and atom["args"][1] not in orange_labels
    }
    if len(destinations) == 1:
        return destinations.pop()
    if len(destinations) > 1:
        raise OrangeGroundingError(
            f"The language model proposed multiple destination surfaces: {sorted(destinations)}."
        )

    containers = [bbox["label"] for bbox in bboxes if _is_container(bbox.get("label", ""))]
    if len(containers) != 1:
        raise OrangeGroundingError(
            "Could not identify exactly one destination box. "
            f"Detected container labels: {containers or 'none'}."
        )
    return containers[0]


def ground_four_orange_instruction(
    bboxes: list[dict],
    grounded_atoms: list[dict],
    task_instruction: str,
    expected_count: int = 4,
) -> GroundingResult:
    """Ground one selected orange and produce exactly one manipulation goal."""
    boxes_out = deepcopy(bboxes)
    atoms_out = deepcopy(grounded_atoms)
    selector = parse_orange_selector(task_instruction)
    if selector is None:
        return GroundingResult(boxes_out, atoms_out, selected_label=None, selector=None)

    indexed_oranges = [
        (index, bbox, _bbox_center(bbox))
        for index, bbox in enumerate(boxes_out)
        if _is_orange(bbox.get("label", ""))
    ]
    if len(indexed_oranges) != expected_count:
        raise OrangeGroundingError(
            f"Expected exactly {expected_count} oranges for a numbered selection, "
            f"but perception detected {len(indexed_oranges)}. Refusing to plan a grasp."
        )
    if selector.rank > len(indexed_oranges):
        raise OrangeGroundingError(
            f"Selector '{selector.description}' requests rank {selector.rank}, "
            f"but only {len(indexed_oranges)} oranges were detected."
        )

    old_labels = [bbox["label"] for _, bbox, _ in indexed_oranges]
    if len(set(old_labels)) != len(old_labels):
        raise OrangeGroundingError(f"Orange detections do not have unique labels: {old_labels}.")

    left_to_right = sorted(indexed_oranges, key=lambda item: (item[2][0], item[2][1]))
    label_map: dict[str, str] = {}
    for rank, (_, bbox, _) in enumerate(left_to_right, start=1):
        old_label = bbox["label"]
        new_label = f"orange_{rank}"
        bbox["label"] = new_label
        label_map[old_label] = new_label

    orange_labels = set(label_map)
    for atom in atoms_out:
        atom["args"] = [label_map.get(arg, arg) for arg in atom.get("args", [])]

    axis_index = 0 if selector.axis == "x" else 1
    visually_ranked = sorted(indexed_oranges, key=lambda item: (item[2][axis_index], item[2][1 - axis_index]))
    if selector.reverse:
        visually_ranked.reverse()
    selected_old_label = visually_ranked[selector.rank - 1][1]["label"]
    selected_label = label_map.get(selected_old_label, selected_old_label)

    canonical_orange_labels = set(label_map.values())
    non_orange_atoms = [
        atom
        for atom in atoms_out
        if not (
            atom.get("predicate") in {"on", "holding"}
            and atom.get("args")
            and atom["args"][0] in canonical_orange_labels
        )
    ]
    has_on_goal = any(
        atom.get("predicate") == "on"
        and len(atom.get("args", [])) == 2
        and atom["args"][0] in orange_labels
        for atom in grounded_atoms
    )
    has_holding_goal = any(
        atom.get("predicate") == "holding"
        and len(atom.get("args", [])) == 1
        and atom["args"][0] in orange_labels
        for atom in grounded_atoms
    )
    if has_on_goal or _instruction_mentions_container(task_instruction):
        destination = _destination_label(boxes_out, grounded_atoms, orange_labels)
        destination = label_map.get(destination, destination)
        selected_goal = {"predicate": "on", "args": [selected_label, destination]}
    elif has_holding_goal:
        selected_goal = {"predicate": "holding", "args": [selected_label]}
    else:
        raise OrangeGroundingError(
            "The instruction selected an orange but did not provide an unambiguous pick or place goal."
        )
    non_orange_atoms.append(selected_goal)
    return GroundingResult(boxes_out, non_orange_atoms, selected_label=selected_label, selector=selector)

