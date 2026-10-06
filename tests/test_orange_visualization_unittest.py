"""Dependency-light tests for the four-orange operator visualization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tiptop.orange_demo import ground_four_orange_instruction
from tiptop.orange_visualization import (
    orange_grounding_metadata,
    render_orange_grounding_overlay,
    save_orange_grounding_visualization,
)


def scene() -> tuple[list[dict], list[dict]]:
    bboxes = [
        {"box_2d": [300, 700, 500, 800], "label": "orange_d"},
        {"box_2d": [300, 100, 500, 200], "label": "orange_a"},
        {"box_2d": [100, 820, 700, 990], "label": "box"},
        {"box_2d": [310, 500, 510, 600], "label": "orange_c"},
        {"box_2d": [320, 300, 520, 400], "label": "orange_b"},
    ]
    atoms = [{"predicate": "on", "args": ["orange_a", "box"]}]
    return bboxes, atoms


class OrangeVisualizationTests(unittest.TestCase):
    def setUp(self) -> None:
        bboxes, atoms = scene()
        self.result = ground_four_orange_instruction(bboxes, atoms, "put orange 2 in the box")
        self.rgb = np.full((600, 1000, 3), 242, dtype=np.uint8)

    def test_overlay_marks_selected_orange_in_green(self) -> None:
        overlay = render_orange_grounding_overlay(self.rgb, self.result)

        self.assertEqual(overlay.shape, self.rgb.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        selected_border = overlay[192, 300]
        self.assertGreater(int(selected_border[1]), 170)
        self.assertLess(int(selected_border[0]), 100)

    def test_metadata_records_selector_and_final_goal(self) -> None:
        metadata = orange_grounding_metadata("put orange 2 in the box", self.result)

        self.assertEqual(metadata["selected_label"], "orange_2")
        self.assertEqual(metadata["selector"]["rank"], 2)
        self.assertEqual(
            metadata["grounded_atoms"],
            [{"predicate": "on", "args": ["orange_2", "box"]}],
        )

    def test_save_writes_overlay_and_json_record(self) -> None:
        overlay = render_orange_grounding_overlay(self.rgb, self.result)
        metadata = orange_grounding_metadata("put orange 2 in the box", self.result)

        with tempfile.TemporaryDirectory() as temp_dir:
            overlay_path, metadata_path = save_orange_grounding_visualization(
                Path(temp_dir),
                overlay,
                metadata,
            )
            self.assertTrue(overlay_path.is_file())
            self.assertTrue(metadata_path.is_file())
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["selected_label"], "orange_2")

    def test_pixel_coordinate_boxes_are_supported(self) -> None:
        for bbox in self.result.bboxes:
            ymin, xmin, ymax, xmax = bbox["box_2d"]
            bbox["box_2d_pixels"] = [xmin, ymin * 0.6, xmax, ymax * 0.6]

        overlay = render_orange_grounding_overlay(self.rgb, self.result)
        self.assertEqual(overlay.shape, self.rgb.shape)


if __name__ == "__main__":
    unittest.main()
