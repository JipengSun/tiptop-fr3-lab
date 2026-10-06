"""Tests for bounding-box coordinate parsing."""

from tiptop.perception.bbox_utils import (
    apply_gemini_norm_bboxes,
    normalize_openai_bboxes,
    parse_openai_box_2d,
)


def test_apply_gemini_norm_bboxes():
    bboxes = [{"label": "box", "box_2d": [100, 50, 800, 200]}]
    apply_gemini_norm_bboxes(bboxes, 1280, 720)
    assert bboxes[0]["box_2d_format"] == "gemini_norm"
    assert bboxes[0]["box_2d_pixels"] == [64, 72, 256, 576]


def test_parse_keeps_true_pixel_coords_on_left():
    raw = [10, 50, 180, 680]
    _, pixels, fmt = parse_openai_box_2d(raw, 1280, 720)
    assert fmt == "pixel_xyxy"
    assert pixels == list(raw)


def test_parse_uses_pixels_when_coord_exceeds_1000():
    raw = [1050, 80, 1200, 550]
    _, pixels, fmt = parse_openai_box_2d(raw, 1280, 720)
    assert fmt == "pixel_xyxy"
    assert pixels == list(raw)


def test_normalize_legacy_openai_ambiguous_prefers_gemini_joint():
    """Legacy json_object path: ambiguous 0-1000 values on HD images."""
    bboxes = [
        {"box_2d": [400, 80, 800, 550], "label": "box"},
        {"box_2d": [580, 420, 640, 480], "label": "peach_1"},
        {"box_2d": [660, 450, 720, 510], "label": "peach_2"},
    ]
    normalize_openai_bboxes(bboxes, 1280, 720)
    assert bboxes[0]["box_2d_format"] == "gemini_norm"
    assert bboxes[0]["box_2d_pixels"] == [102, 288, 704, 576]
