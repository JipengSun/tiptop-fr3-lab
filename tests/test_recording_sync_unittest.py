"""What the shared timestamp grid does when one camera dies mid-run.

A grab loop that raises ends that camera's SVO where it stopped; the others keep recording to
the end of the motion. The grid has to span the union of those windows, because clipping it to
the earliest ending throws away footage that is sitting, intact, in the healthy cameras'
.svo2 masters -- which is exactly what happened to a 35 s external view that reached disk as
4.9 s of .mp4.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from tiptop.perception.cameras.zed_camera import convert_svos_to_mp4_synced

MODULE = "tiptop.perception.cameras.zed_camera"


def ticks(first_ms: int, n: int, period_ms: int) -> list[int]:
    return [first_ms + i * period_ms for i in range(n)]


def ticks_until(first_ms: int, last_ms: int, period_ms: int) -> list[int]:
    """A stream that runs from `first_ms` to exactly `last_ms` -- a camera that stayed alive."""
    stamps = list(range(first_ms, last_ms, period_ms))
    return stamps + [last_ms]


class SyncedWindowTests(unittest.TestCase):
    def sync(self, streams: dict[str, list[int]]) -> tuple[dict, dict]:
        """Run the sync over fake timestamp lists; return (metadata, grid each camera got)."""
        items = [(Path(f"{name}.svo2"), Path(f"{name}.mp4")) for name in streams]
        grids: dict[str, list[int]] = {}

        def fake_read(svo_path):
            return streams[Path(svo_path).stem]

        def fake_convert(svo_path, mp4_path, sample_times_ms, frame_timestamps_ms, fps, crf=20):
            grids[Path(mp4_path).stem] = list(sample_times_ms)

        with mock.patch(f"{MODULE}.read_svo_timestamps", side_effect=fake_read), \
             mock.patch(f"{MODULE}.convert_svo_to_mp4_resampled", side_effect=fake_convert):
            meta = convert_svos_to_mp4_synced(items, output_fps=30)
        return meta, grids

    def test_healthy_stream_keeps_its_full_length(self) -> None:
        """The camera that ran to the end is not cut back to the one that died at 5 s."""
        meta, grids = self.sync(
            {
                "external_cam": ticks(1_000, 1051, 33),  # ~34.7 s at 30 fps
                "hand_cam": ticks(1_000, 293, 16),       # ~4.7 s, then the grab loop died
            }
        )
        self.assertEqual(meta["t_end_ms"], 1_000 + 1050 * 33)
        self.assertAlmostEqual(meta["n_frames"] / 30, 34.7, delta=0.2)
        # Both .mp4 files stay on one grid, so frame i is the same instant in each.
        self.assertEqual(grids["external_cam"], grids["hand_cam"])

    def test_short_stream_is_marked_as_held(self) -> None:
        """The dead camera's tail is reported as padding, not passed off as footage."""
        meta, _ = self.sync(
            {
                "external_cam": ticks(1_000, 1051, 33),
                "hand_cam": ticks(1_000, 293, 16),
            }
        )
        hand = meta["cameras"]["hand_cam"]
        external = meta["cameras"]["external_cam"]
        self.assertEqual(hand["n_real_frames"] + hand["held_tail_frames"], meta["n_frames"])
        self.assertGreater(hand["held_tail_frames"], 0)
        self.assertAlmostEqual(hand["ended_early_ms"] / 1000, 30.0, delta=0.5)
        self.assertEqual(external["held_tail_frames"], 0)
        self.assertEqual(external["ended_early_ms"], 0)

    def test_healthy_run_has_no_padding(self) -> None:
        """The ordinary case is unchanged: two live cameras, no held frames anywhere."""
        meta, _ = self.sync(
            {
                "external_cam": ticks_until(1_000, 30_900, 33),
                "hand_cam": ticks_until(1_010, 30_900, 16),
            }
        )
        for cam in meta["cameras"].values():
            self.assertEqual(cam["held_tail_frames"], 0)
        self.assertAlmostEqual(meta["n_frames"] / 30, 29.6, delta=0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
