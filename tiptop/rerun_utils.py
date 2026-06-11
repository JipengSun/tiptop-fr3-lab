"""Rerun logging helpers for TiPToP perception outputs."""

import logging

import numpy as np
import rerun as rr
from jaxtyping import Float, UInt8

from tiptop.perception.bbox_utils import bbox_to_pixel_xyxy

_log = logging.getLogger(__name__)


def log_camera_to_rerun(
    intrinsics: Float[np.ndarray, "3 3"],
    world_from_cam: Float[np.ndarray, "4 4"],
    entity_path: str = "cam",
) -> None:
    """Log camera intrinsics and world pose for Rerun pinhole projection."""
    rr.log(entity_path, rr.Pinhole(image_from_camera=intrinsics))
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=world_from_cam[:3, 3],
            mat3x3=world_from_cam[:3, :3],
            axis_length=0.05,
        ),
    )


def log_bbox_detections_to_rerun(
    bboxes: list[dict],
    img_width: int,
    img_height: int,
    entity_path: str = "cam/bboxes",
) -> None:
    """Log 2D boxes aligned with the pinhole camera (pixel XYXY)."""
    xyxy = []
    labels = []
    for bbox in bboxes:
        box = bbox.get("box_2d", [])
        coords = bbox_to_pixel_xyxy(bbox, img_width, img_height)
        if coords is None:
            continue
        xyxy.append(coords)
        labels.append(bbox.get("label", f"obj_{len(xyxy)}"))
    if len(xyxy) == 0:
        return
    rr.log(
        entity_path,
        rr.Boxes2D(
            array=xyxy,
            array_format=rr.Box2DFormat.XYXY,
            labels=labels[: len(xyxy)],
            show_labels=True,
        ),
    )


def log_perception_to_rerun(
    rgb: UInt8[np.ndarray, "h w 3"],
    intrinsics: Float[np.ndarray, "3 3"],
    world_from_cam: Float[np.ndarray, "4 4"],
    depth_map: Float[np.ndarray, "h w"],
    bboxes: list[dict],
    bbox_viz: np.ndarray | None,
    masks_viz: np.ndarray | None,
    xyz_map: Float[np.ndarray, "h w 3"],
    rgb_map: Float[np.ndarray, "h w 3"],
) -> None:
    """Log perception outputs under a single camera entity tree for correct 2D/3D alignment."""
    log_camera_to_rerun(intrinsics, world_from_cam)

    rr.log("cam/rgb", rr.Image(rgb))
    rr.log("cam/depth", rr.DepthImage(depth_map, meter=1.0))

    log_bbox_detections_to_rerun(bboxes, rgb.shape[1], rgb.shape[0])

    if bbox_viz is not None:
        rr.log("cam/bboxes_viz", rr.Image(bbox_viz))
    if masks_viz is not None:
        rr.log("cam/masks_viz", rr.Image(masks_viz))

    rr.log(
        "pcd",
        rr.Points3D(
            positions=xyz_map.reshape(-1, 3),
            colors=rgb_map.reshape(-1, 3),
        ),
    )
