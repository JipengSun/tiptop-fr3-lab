import json
import logging
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Generator

import cv2
import dill
import numpy as np
import open3d as o3d
import torch
from jaxtyping import Bool, Float, UInt8
from PIL import Image

import tiptop
from tiptop.config import get_tiptop_cfg_path
from tiptop.perception.cameras.zed_camera import ZedCamera, convert_svo_to_mp4
from tiptop.perception.utils import get_o3d_pcd
from tiptop.perception.visualization import visualize_detections, visualize_masks
from tiptop.utils import NumpyEncoder

_log = logging.getLogger(__name__)


def save_rgb_image(rgb: UInt8[np.ndarray, "h w 3"], path: Path) -> None:
    """Save RGB camera frame as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_bbox_viz_image(bbox_viz: np.ndarray, path: Path) -> None:
    """Save bounding-box overlay image as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(bbox_viz, cv2.COLOR_RGB2BGR))


def save_mask_viz_image(masks_viz: np.ndarray, path: Path) -> None:
    """Save segmentation mask overlay image as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(masks_viz, cv2.COLOR_RGB2BGR))


def save_depth_colormap(depth_map: Float[np.ndarray, "h w"], path: Path) -> np.ndarray:
    """Save a turbo colormap visualization of the depth map."""
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = depth_map > 0
    if not valid.any():
        colormap = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    else:
        d_norm = np.zeros_like(depth_map, dtype=np.float32)
        d_min, d_max = depth_map[valid].min(), depth_map[valid].max()
        d_norm[valid] = (depth_map[valid] - d_min) / (d_max - d_min + 1e-9)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(d_uint8, cv2.COLORMAP_TURBO)
        colormap = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        colormap[~valid] = 0
    cv2.imwrite(str(path), cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR))
    return colormap


def save_scene_3d_image(pcd: o3d.geometry.PointCloud, path: Path, width: int = 960, height: int = 720) -> np.ndarray | None:
    """Render the point cloud to a PNG snapshot for offline inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(pcd.points) == 0:
        _log.warning("Skipping scene_3d.png: empty point cloud")
        return None
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
        vis.reset_view_point(True)
        vis.poll_events()
        vis.update_renderer()
        image = np.asarray(vis.capture_screen_float_buffer(True))
        vis.destroy_window()
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))
        return image_uint8
    except Exception:
        _log.warning("Failed to render scene_3d.png", exc_info=True)
        return None


@cache
def _get_git_root() -> Path:
    """Return the repository root."""
    return Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).parent,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip())


def _collect_git_info() -> dict:
    """Return git commit hash and dirty status for metadata.

    pixi.lock is excluded from the dirty check as it's not relevant for debugging
    """
    try:
        root = _get_git_root()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)pixi.lock"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(porcelain.strip())
        return {"commit": commit, "dirty": dirty, "porcelain": porcelain.strip() if dirty else None}
    except (FileNotFoundError, subprocess.CalledProcessError):
        _log.warning("Failed to collect git info", exc_info=True)
        return {"commit": None, "dirty": None, "porcelain": None}


def _get_git_diff() -> str | None:
    """Return the full git diff against HEAD, excluding pixi.lock, or None if unavailable or empty."""
    try:
        root = _get_git_root()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", ".", ":(exclude)pixi.lock"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return diff if diff else None
    except (FileNotFoundError, subprocess.CalledProcessError):
        _log.warning("Failed to get git diff", exc_info=True)
        return None


@contextmanager
def record_cameras(recordings: list[tuple[ZedCamera, Path, Path | None]]) -> Generator[None, None, None]:
    """Context manager for recording multiple ZED cameras simultaneously.

    All cameras stop collecting frames at the same time on exit, then MP4
    conversion runs sequentially afterwards.

    Args:
        recordings: List of (camera, svo_path, mp4_path) tuples
    """
    stop_events: list[threading.Event] = []
    threads: list[threading.Thread] = []

    for camera, svo_path, _ in recordings:
        stop_event = threading.Event()

        def recording_loop(cam=camera, event=stop_event):
            while not event.is_set():
                try:
                    cam.read_camera()
                except Exception as e:
                    _log.error(f"Error grabbing frame during recording: {e}")
                    break

        camera.start_recording(str(svo_path))
        thread = threading.Thread(target=recording_loop)
        thread.start()
        _log.info(f"Started recording camera {camera.serial} to {svo_path}")
        stop_events.append(stop_event)
        threads.append(thread)

    try:
        yield
    finally:
        # Signal all cameras to stop simultaneously so recordings are the same length
        for event in stop_events:
            event.set()
        for (camera, svo_path, _), thread in zip(recordings, threads):
            thread.join(timeout=3.0)
            camera.stop_recording()
            _log.info(f"Stopped recording camera {camera.serial}")

        # Convert to MP4 after all cameras have stopped
        for camera, svo_path, mp4_path in recordings:
            if mp4_path is None:
                continue
            actual_svo_path = svo_path
            if not svo_path.exists():
                svo2_path = svo_path.with_suffix(".svo2")
                if svo2_path.exists():
                    _log.debug(f"SVO actually written by ZED SDK to {svo2_path.name}")
                    actual_svo_path = svo2_path
            if actual_svo_path.exists():
                convert_svo_to_mp4(actual_svo_path, mp4_path)
            else:
                raise FileNotFoundError(
                    f"Recording failed: SVO file not found at {svo_path} or {svo_path.with_suffix('.svo2')}"
                )


def save_perception_outputs(
    rgb: UInt8[np.ndarray, "h w 3"],
    intrinsics_matrix: Float[np.ndarray, "3 3"],
    depth_map: Float[np.ndarray, "h w"],
    xyz_map: Float[np.ndarray, "n 3"],
    rgb_map: Float[np.ndarray, "n 3"],
    bboxes: list[dict],
    masks: np.ndarray,
    save_dir: Path,
    gripper_mask: Bool[np.ndarray, "h w"] | None = None,
):
    """Save perception outputs to disk.

    Saves on every run:
      - rgb.png, bboxes_viz.png, masks_viz.png (top-level for quick access)
      - perception/rgb.png, depth_colormap.png, scene_3d.png, depth.png, pointcloud.ply, etc.
    """
    start_time = time.perf_counter()
    save_dir.mkdir(parents=True, exist_ok=True)
    perception_dir = save_dir / "perception"
    perception_dir.mkdir(exist_ok=True)

    save_rgb_image(rgb, save_dir / "rgb.png")
    save_rgb_image(rgb, perception_dir / "rgb.png")

    # Intrinsics
    intrinsics_dict = {"intrinsics": intrinsics_matrix.tolist()}
    with open(perception_dir / "intrinsics.json", "w") as f:
        json.dump(intrinsics_dict, f, indent=2)

    # Convert depth from meters to millimeters for uint16 storage
    depth_mm = depth_map * 1000.0
    depth_mm = np.clip(depth_mm, 0, 65535)
    depth_uint16 = depth_mm.astype(np.uint16)
    cv2.imwrite(str(perception_dir / "depth.png"), depth_uint16)
    save_depth_colormap(depth_map, perception_dir / "depth_colormap.png")

    # Create point cloud and write
    pcd = get_o3d_pcd(xyz_map, rgb_map)
    o3d.io.write_point_cloud(str(perception_dir / "pointcloud.ply"), pcd)
    save_scene_3d_image(pcd, perception_dir / "scene_3d.png")

    # Write bboxes
    rgb_pil = Image.fromarray(rgb)
    bbox_viz = visualize_detections(rgb_pil, bboxes, output_path=None, show_plot=False)
    if bbox_viz is not None:
        save_bbox_viz_image(bbox_viz, save_dir / "bboxes_viz.png")
        save_bbox_viz_image(bbox_viz, perception_dir / "bboxes_viz.png")
    else:
        bbox_viz = rgb.copy()
    with open(perception_dir / "bboxes.json", "w") as f:
        json.dump(bboxes, f, indent=2)

    # Write masks
    masks_viz = visualize_masks(rgb_pil, masks, bboxes)
    save_mask_viz_image(masks_viz, save_dir / "masks_viz.png")
    save_mask_viz_image(masks_viz, perception_dir / "masks_viz.png")
    masks_bool = masks > 0.5
    np.savez_compressed(str(perception_dir / "masks.npz"), masks_bool)  # masks are sparse so can compress

    if gripper_mask is not None:
        gripper_mask_img = Image.fromarray(gripper_mask.astype(np.uint8) * 255)
        gripper_mask_img.save(str(perception_dir / "gripper_mask.png"))

    save_dur = time.perf_counter() - start_time
    _log.info(f"Saved perception outputs to {save_dir} in {save_dur:.2f}s")
    return bbox_viz, masks_viz


def save_run_outputs(save_dir: Path, env, grasps: dict) -> None:
    """Save cuTAMP environment, grasps, and run artifacts (config) to disk."""
    # Save cutamp environment and grasps
    perception_dir = save_dir / "perception"
    perception_dir.mkdir(parents=True, exist_ok=True)
    with open(perception_dir / "cutamp_env.pkl", "wb") as f:
        dill.dump(env, f)
    _log.info(f"Saved cutamp env to {perception_dir}/cutamp_env.pkl")

    torch.save(grasps, perception_dir / "grasps.pt")
    _log.info(f"Saved grasps to {perception_dir}/grasps.pt")

    # tiptop config for reproducibility
    shutil.copy2(get_tiptop_cfg_path(), save_dir / "tiptop.yml")
    _log.info(f"Saved tiptop config to {save_dir}/tiptop.yml")


def save_run_metadata(
    save_dir: Path,
    timestamp: str,
    task_instruction: str | None,
    q_at_capture: np.ndarray | list | None,
    world_from_cam: np.ndarray | list | None,
    perception_duration: float | None,
    grounded_atoms: list[dict] | None,
    planning_success: bool | None,
    planning_failure_reason: str | None,
    planning_duration: float | None,
) -> None:
    """Save structured run metadata to metadata.json."""
    git_info = _collect_git_info()
    if git_info["dirty"]:
        diff = _get_git_diff()
        if diff:
            (save_dir / "git.diff").write_text(diff, encoding="utf-8")

    metadata = {
        "task_instruction": task_instruction,
        "timestamp": timestamp,
        "observation": {
            "q_at_capture": q_at_capture,
            "world_from_cam": world_from_cam,
        },
        "perception": {
            "grounded_atoms": grounded_atoms,
            "duration": round(perception_duration, 3) if perception_duration is not None else None,
        },
        "planning": {
            "success": planning_success,
            "failure_reason": planning_failure_reason,
            "duration": round(planning_duration, 3) if planning_duration is not None else None,
        },
        "version": "1.0.0",
        "tiptop_version": tiptop.__version__,
        "git": git_info,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    _log.info(f"Saved run metadata to {save_dir}/metadata.json")
