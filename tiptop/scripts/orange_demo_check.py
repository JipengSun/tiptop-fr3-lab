"""Dry-run the deterministic four-orange language grounder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tiptop.orange_demo import OrangeGroundingError, ground_four_orange_instruction
from tiptop.orange_visualization import (
    orange_grounding_metadata,
    render_orange_grounding_overlay,
    save_orange_grounding_visualization,
)


def _fixture_detections() -> tuple[list[dict], list[dict]]:
    bboxes = [
        {"box_2d": [380, 690, 560, 820], "label": "orange_right"},
        {"box_2d": [360, 100, 550, 230], "label": "orange_left"},
        {"box_2d": [120, 820, 680, 990], "label": "cardboard_box"},
        {"box_2d": [370, 490, 555, 620], "label": "orange_middle_right"},
        {"box_2d": [390, 300, 570, 430], "label": "orange_middle_left"},
    ]
    atoms = [{"predicate": "on", "args": ["orange_left", "cardboard_box"]}]
    return bboxes, atoms


def _fixture_image(bboxes: list[dict]) -> np.ndarray:
    image = Image.new("RGB", (1000, 600), color=(241, 245, 249))
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        ymin, xmin, ymax, xmax = bbox["box_2d"]
        coords = (xmin, int(ymin * 0.6), xmax, int(ymax * 0.6))
        if "orange" in bbox["label"]:
            draw.ellipse(coords, fill=(249, 115, 22), outline=(154, 52, 18), width=4)
        else:
            draw.rectangle(coords, fill=(186, 230, 253), outline=(3, 105, 161), width=4)
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve an English or Chinese orange selector without starting robot or GPU services."
    )
    parser.add_argument(
        "instruction",
        help="For example: 'put the third orange in the box' or '把从左数第二个橙子放进盒子'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional directory for orange_selection_viz.png and perception/orange_grounding.json.",
    )
    args = parser.parse_args()
    bboxes, atoms = _fixture_detections()
    try:
        result = ground_four_orange_instruction(bboxes, atoms, args.instruction)
    except OrangeGroundingError as exc:
        parser.error(str(exc))
    if args.output is not None and result.selected_label is not None:
        overlay = render_orange_grounding_overlay(_fixture_image(result.bboxes), result)
        save_orange_grounding_visualization(
            args.output,
            overlay,
            orange_grounding_metadata(args.instruction, result),
        )
    print(
        json.dumps(
            {
                "selector": result.selector.description if result.selector else None,
                "selected_label": result.selected_label,
                "bboxes": result.bboxes,
                "predicates": result.grounded_atoms,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

