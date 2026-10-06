import logging
from typing import Awaitable, Callable, Protocol

import aiohttp
import numpy as np

from tiptop.config import tiptop_cfg

_log = logging.getLogger(__name__)
from tiptop.perception.cameras.frame import Frame
from tiptop.perception.cameras.rs_camera import (
    RealsenseCamera,
    RealsenseFrame,
    rs_infer_depth_async,
)
from tiptop.perception.cameras.zed_camera import (
    ZedCamera,
    ZedFrame,
    zed_infer_depth_async,
)

# Callable that takes an aiohttp session and a camera frame and returns a float depth map (H, W) in metres.
DepthEstimator = Callable[[aiohttp.ClientSession, Frame], Awaitable[np.ndarray]]


class Camera(Protocol):
    serial: str

    def read_camera(self) -> Frame: ...
    def close(self) -> None: ...


def get_depth_estimator(cam: Camera) -> DepthEstimator:
    """Get the appropriate FoundationStereo depth estimator for the given camera. Call once per camera."""
    if isinstance(cam, ZedCamera):
        intrinsics = cam.get_intrinsics()

        async def _zed_estimate(session: aiohttp.ClientSession, f: Frame) -> np.ndarray:
            return await zed_infer_depth_async(session, f, intrinsics)  # type: ignore[arg-type]

        return _zed_estimate
    elif isinstance(cam, RealsenseCamera):
        intrinsics = cam.get_intrinsics()

        async def _rs_estimate(session: aiohttp.ClientSession, f: Frame) -> np.ndarray:
            return await rs_infer_depth_async(session, f, intrinsics)  # type: ignore[arg-type]

        return _rs_estimate
    else:
        raise ValueError(f"No depth estimator available for camera type: {type(cam).__name__}")


def _get_zed_camera(cam_cfg, depth: bool = False, pointcloud: bool = False) -> ZedCamera:
    """Create a ZedCamera from config."""
    serial = str(cam_cfg.serial)
    flip = cam_cfg.get("flip", False)
    resolution = cam_cfg.get("resolution", "HD720")
    fps = cam_cfg.get("fps", 60)
    return ZedCamera(serial, resolution=resolution, fps=fps, flip=flip, depth=depth, pointcloud=pointcloud)


def get_hand_camera(depth: bool = False) -> Camera:
    """Get the hand camera by serial number."""
    cfg = tiptop_cfg()
    cam_cfg = cfg.cameras.hand
    cam_type = cam_cfg.type
    if cam_type == "zed":
        return _get_zed_camera(cam_cfg, depth=depth)
    elif cam_type == "realsense":
        return RealsenseCamera(str(cam_cfg.serial), enable_depth=depth)
    else:
        raise ValueError(f"Unknown camera type: {cam_type}")


def get_external_camera() -> Camera | None:
    """Get the external camera by serial number, or None if not configured / unavailable.

    TiPToP perception uses only the hand camera; external is optional (recording / viz-scene).
    """
    cfg = tiptop_cfg()
    cam_cfg = cfg.cameras.get("external")
    if cam_cfg is None:
        return None

    serial = str(cam_cfg.get("serial", "")).strip()
    if not serial or serial in {"external-serial", "none", "null"}:
        _log.info("No external camera configured — skipping")
        return None

    cam_type = cam_cfg.type
    try:
        if cam_type == "zed":
            return _get_zed_camera(cam_cfg)
        elif cam_type == "realsense":
            return RealsenseCamera(serial)
        else:
            raise ValueError(f"Unknown camera type: {cam_type}")
    except RuntimeError as e:
        _log.warning(
            "External camera %s failed to open (%s) — continuing without it. "
            "Perception uses the hand camera only. Plug in the external ZED or set "
            "cameras.external.serial to empty in tiptop.yml.",
            serial,
            e,
        )
        return None
