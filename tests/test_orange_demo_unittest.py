"""Dependency-free tests for the four-orange grounding extension."""

from __future__ import annotations

import unittest
from copy import deepcopy

from tiptop.orange_demo import OrangeGroundingError, ground_four_orange_instruction, parse_orange_selector


def scene() -> tuple[list[dict], list[dict]]:
    bboxes = [
        {"box_2d": [300, 700, 500, 800], "label": "orange_d"},
        {"box_2d": [300, 100, 500, 200], "label": "orange_a"},
        {"box_2d": [100, 820, 700, 990], "label": "box"},
        {"box_2d": [310, 500, 510, 600], "label": "orange_c"},
        {"box_2d": [320, 300, 520, 400], "label": "orange_b"},
    ]
    atoms = [
        {"predicate": "on", "args": ["orange_a", "box"]},
        {"predicate": "on", "args": ["orange_b", "box"]},
    ]
    return bboxes, atoms


class OrangeDemoTests(unittest.TestCase):
    def test_supported_selectors(self) -> None:
        cases = (
            ("put orange 3 in the box", "orange_3"),
            ("put the second orange from the right in the box", "orange_3"),
            ("put the left orange in the box", "orange_1"),
            ("把从右数第二个橙子放进盒子", "orange_3"),
            ("把第4个橙子抓到盒子里", "orange_4"),
            ("2", "orange_2"),
        )
        for instruction, selected in cases:
            with self.subTest(instruction=instruction):
                bboxes, atoms = scene()
                result = ground_four_orange_instruction(bboxes, atoms, instruction)
                self.assertEqual(result.selected_label, selected)
                self.assertEqual(result.grounded_atoms, [{"predicate": "on", "args": [selected, "box"]}])

    def test_pick_only_instruction_remains_a_holding_goal(self) -> None:
        bboxes, _ = scene()
        atoms = [{"predicate": "holding", "args": ["orange_a"]}]
        result = ground_four_orange_instruction(bboxes, atoms, "pick up orange 2")

        self.assertEqual(result.grounded_atoms, [{"predicate": "holding", "args": ["orange_2"]}])

    def test_topmost_uses_vertical_position(self) -> None:
        bboxes, atoms = scene()
        bboxes[3]["box_2d"] = [20, 500, 220, 600]
        result = ground_four_orange_instruction(bboxes, atoms, "put the top orange in the box")
        self.assertEqual(result.selected_label, "orange_3")

    def test_unrecognized_selector_is_passthrough(self) -> None:
        bboxes, atoms = scene()
        result = ground_four_orange_instruction(bboxes, atoms, "put an orange in the box")
        self.assertIsNone(result.selected_label)
        self.assertEqual(result.bboxes, bboxes)
        self.assertEqual(result.grounded_atoms, atoms)

    def test_inputs_are_not_mutated(self) -> None:
        bboxes, atoms = scene()
        bboxes_before, atoms_before = deepcopy(bboxes), deepcopy(atoms)
        ground_four_orange_instruction(bboxes, atoms, "put orange 2 in the box")
        self.assertEqual(bboxes, bboxes_before)
        self.assertEqual(atoms, atoms_before)

    def test_missing_orange_fails_closed(self) -> None:
        bboxes, atoms = scene()
        bboxes.pop(0)
        with self.assertRaisesRegex(OrangeGroundingError, "Expected exactly 4 oranges"):
            ground_four_orange_instruction(bboxes, atoms, "put orange 2 in the box")

    def test_ambiguous_destination_fails_closed(self) -> None:
        bboxes, _ = scene()
        bboxes.append({"box_2d": [100, 0, 700, 80], "label": "blue_bin"})
        with self.assertRaisesRegex(OrangeGroundingError, "exactly one destination"):
            ground_four_orange_instruction(bboxes, [], "put orange 2 in the box")

    def test_chinese_relative_selector(self) -> None:
        selector = parse_orange_selector("请把左边第一个橙子放进盒子")
        self.assertIsNotNone(selector)
        self.assertEqual((selector.axis, selector.rank, selector.reverse), ("x", 1, False))


if __name__ == "__main__":
    unittest.main()

