"""Modified from https://github.com/droid-dataset/droid/blob/main/droid/camera_utils/camera_readers/zed_camera.py"""

import logging
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import aiohttp
import cv2
import numpy as np
from jaxtyping import Float, UInt8
from tqdm import tqdm

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras.frame import Frame

_log = logging.getLogger(__name__)


def time_ms():
    return time.time_ns() // 1_000_000


@dataclass(frozen=True, kw_only=True)
class ZedFrame(Frame):
    """Frame from ZED which also includes raw stereo pair and point cloud."""

    left_bgra: UInt8[np.ndarray, "h w 4"]  # BGRA left uint8
    right_bgra: UInt8[np.ndarray, "h w 4"]  # BGRA right uint8
    pointcloud: Float[np.ndarray, "h w 4"] | None = None  # Hardware pointcloud float32 millimeters


@dataclass(frozen=True)
class ZedIntrinsics:
    """Intrinsics for ZED camera."""

    K_left: Float[np.ndarray, "3 3"]  # Left RGB camera matrix
    K_right: Float[np.ndarray, "3 3"]  # Right RGB camera matrix
    distortion_left: Float[np.ndarray, "12"]
    distortion_right: Float[np.ndarray, "12"]
    baseline: float  # Meters


def _custom_params(resolution: str, fps: int, flip: bool = False):
    """Build camera init params from explicit resolution and fps strings."""
    import pyzed.sl as sl

    resolution_map = {
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD2K": sl.RESOLUTION.HD2K,
    }
    if resolution not in resolution_map:
        raise ValueError(f"Unknown resolution '{resolution}', expected one of {list(resolution_map)}")

    return dict(
        depth_minimum_distance=0.1,
        camera_resolution=resolution_map[resolution],
        depth_stabilization=False,
        camera_fps=fps,
        camera_image_flip=sl.FLIP_MODE.ON if flip else sl.FLIP_MODE.OFF,
    )


class ZedCamera:
    def __init__(
        self,
        serial: str,
        depth: bool = False,
        pointcloud: bool = False,
        resolution: str = "HD720",
        fps: int = 60,
        flip: bool = False,
    ):
        import pyzed.sl as sl

        start_time = time.perf_counter()
        self.serial = serial
        self._enable_depth = depth
        self._enable_pcd = pointcloud
        self._is_recording = False

        # Initialize readers
        self._cam = sl.Camera()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._depth = sl.Mat()
        self._pcd = sl.Mat()
        self._runtime = sl.RuntimeParameters()

        # Open the camera
        _log.info(f"Opening ZED camera {self.serial}")
        self._params = _custom_params(resolution, fps, flip=flip)
        sl_params = sl.InitParameters(**self._params)
        sl_params.set_from_serial_number(int(self.serial))
        status = self._cam.open(sl_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera (s/n: {self.serial}) failed to open")

        # Cache the intrinsics
        _ = self.get_intrinsics()
        init_dur = time.perf_counter() - start_time
        _log.info(f"ZED camera (s/n: {self.serial}) initialization complete, took {init_dur:.2f}s")

    @cache
    def get_intrinsics(self) -> ZedIntrinsics:
        calib_params = self._cam.get_camera_information().camera_configuration.calibration_parameters
        baseline_mm = calib_params.get_camera_baseline()
        baseline = baseline_mm / 1000.0

        left = calib_params.left_cam
        K_left = np.array([[left.fx, 0.0, left.cx], [0.0, left.fy, left.cy], [0.0, 0.0, 1.0]])
        distortion_left = np.array(left.disto)

        right = calib_params.right_cam
        K_right = np.array([[right.fx, 0.0, right.cx], [0.0, right.fy, right.cy], [0.0, 0.0, 1.0]])
        distortion_right = np.array(right.disto)

        return ZedIntrinsics(
            K_left=K_left,
            K_right=K_right,
            distortion_left=distortion_left,
            distortion_right=distortion_right,
            baseline=baseline,
        )

    def read_camera(self) -> ZedFrame:
        import pyzed.sl as sl

        # Grab a frame
        err = self._cam.grab(self._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to grab frame from ZED camera {self.serial}")

        # Retrieve data
        timestamp = self._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT)
        left_bgra = self._left_img.get_data()
        self._cam.retrieve_image(self._right_img, sl.VIEW.RIGHT)
        right_bgra = self._right_img.get_data()

        depth = None
        if self._enable_depth:
            self._cam.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
            depth = (self._depth.get_data() / 1000.0).astype(np.float32)  # mm to m

        pointcloud = None
        if self._enable_pcd:
            self._cam.retrieve_measure(self._pcd, sl.MEASURE.XYZ)
            pointcloud = self._pcd.get_data()

        return ZedFrame(
            serial=self.serial,
            timestamp=timestamp,
            rgb=cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2RGB),
            intrinsics=self.get_intrinsics().K_left,
            depth=depth,
            left_bgra=left_bgra,
            right_bgra=right_bgra,
            pointcloud=pointcloud,
        )

    def start_recording(self, filename: str):
        """Start recording camera stream to SVO file using H265 compression."""
        import pyzed.sl as sl

        if not filename.endswith(".svo"):
            raise ValueError("Recording filename must end with .svo")

        recording_param = sl.RecordingParameters(filename, sl.SVO_COMPRESSION_MODE.H265)
        err = self._cam.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable recording to {filename}")
        self._is_recording = True
        _log.info(f"Started recording to {filename}")

    def stop_recording(self):
        """Stop recording camera stream."""
        if self._is_recording:
            self._cam.disable_recording()
            self._is_recording = False
            _log.info("Stopped recording")

    def close(self):
        try:
            self.stop_recording()
            self._cam.close()
        except Exception as e:
            _log.warning(f"Error closing ZED camera {self.serial}: {e}")


def zed_infer_depth(
    frame: ZedFrame,
    intrinsics: ZedIntrinsics,
) -> Float[np.ndarray, "h w"]:
    """Estimate depth from Zed frame and intrinsics using FoundationStereo. Synchronous version."""
    from tiptop.perception.foundation_stereo import infer_depth

    cfg = tiptop_cfg()
    K_left = intrinsics.K_left
    depth = infer_depth(
        cfg.perception.foundation_stereo.url,
        left_rgb=cv2.cvtColor(frame.left_bgra, cv2.COLOR_BGRA2RGB),
        right_rgb=cv2.cvtColor(frame.right_bgra, cv2.COLOR_BGRA2RGB),
        fx=K_left[0, 0],
        fy=K_left[1, 1],
        cx=K_left[0, 2],
        cy=K_left[1, 2],
        baseline=intrinsics.baseline,
    )
    return depth


async def zed_infer_depth_async(
    session: aiohttp.ClientSession,
    frame: ZedFrame,
    intrinsics: ZedIntrinsics,
) -> Float[np.ndarray, "h w"]:
    """Estimate depth from Zed frame and intrinsics using FoundationStereo. Async version."""
    from tiptop.perception.foundation_stereo import infer_depth_async

    cfg = tiptop_cfg()
    K_left = intrinsics.K_left
    depth = await infer_depth_async(
        session,
        cfg.perception.foundation_stereo.url,
        left_rgb=cv2.cvtColor(frame.left_bgra, cv2.COLOR_BGRA2RGB),
        right_rgb=cv2.cvtColor(frame.right_bgra, cv2.COLOR_BGRA2RGB),
        fx=K_left[0, 0],
        fy=K_left[1, 1],
        cx=K_left[0, 2],
        cy=K_left[1, 2],
        baseline=intrinsics.baseline,
    )
    return depth


def read_svo_timestamps(svo_path: Path) -> list[int]:
    """Return IMAGE timestamps (ms) for every frame in an SVO file."""
    import pyzed.sl as sl

    if not svo_path.exists():
        raise FileNotFoundError(f"SVO file not found: {svo_path}")

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.svo_real_time_mode = False
    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO file: {err}")

    rt_param = sl.RuntimeParameters()
    timestamps: list[int] = []
    try:
        while True:
            err = zed.grab(rt_param)
            if err == sl.ERROR_CODE.SUCCESS:
                timestamps.append(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds())
            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            else:
                raise RuntimeError(f"Failed to grab frame at position {zed.get_svo_position()}: {err}")
    finally:
        zed.close()
    return timestamps


def convert_svo_to_mp4_resampled(
    svo_path: Path,
    mp4_path: Path,
    sample_times_ms: list[int],
    frame_timestamps_ms: list[int],
    output_fps: int,
    crf: int = 20,
) -> None:
    """Convert SVO to MP4 by resampling frames onto a shared timestamp grid."""
    import shutil
    import subprocess

    import pyzed.sl as sl

    if not frame_timestamps_ms:
        raise RuntimeError(f"SVO has no frames: {svo_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH, required for SVO to MP4 conversion")

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.svo_real_time_mode = False
    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO file: {err}")

    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(output_fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    left_image = sl.Mat()
    rt_param = sl.RuntimeParameters()
    frame_idx = 0
    try:
        if zed.grab(rt_param) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to read first frame from {svo_path}")

        with tqdm(total=len(sample_times_ms), desc=f"Syncing {svo_path.name}", unit="frame") as pbar:
            for target_t in sample_times_ms:
                while frame_idx < len(frame_timestamps_ms) - 1 and frame_timestamps_ms[frame_idx + 1] <= target_t:
                    frame_idx += 1
                    if zed.grab(rt_param) != sl.ERROR_CODE.SUCCESS:
                        break

                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                frame_bgr = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2BGR)
                try:
                    proc.stdin.write(frame_bgr.tobytes())
                except BrokenPipeError:
                    break
                pbar.update(1)
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read()
        proc.wait()
        zed.close()

    if proc.returncode != 0:
        stderr_msg = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {stderr_msg}")


# How far behind the last camera a stream may stop before it is called a failure rather than
# jitter. Two healthy ZEDs stop within ~30 ms of each other; a grab loop that died mid-run is
# out by tens of seconds. A second sits well clear of both.
EARLY_STOP_WARN_MS = 1000


def convert_svos_to_mp4_synced(
    items: list[tuple[Path, Path]],
    output_fps: int = 30,
    crf: int = 20,
) -> dict:
    """Convert multiple SVO files to MP4, resampled to a shared timestamp grid."""
    if not items:
        return {}

    timestamp_lists = [read_svo_timestamps(svo_path) for svo_path, _ in items]
    if any(not ts for ts in timestamp_lists):
        raise RuntimeError("One or more SVO recordings contain no frames")

    t_start = max(ts[0] for ts in timestamp_lists)
    # Union, not intersection. A camera whose grab loop dies mid-run ends its SVO early, and
    # clipping the grid to min(last) discards every second the *healthy* cameras went on
    # recording -- footage that is sitting in their .svo2 masters and was never lost, only
    # thrown away here. The grid runs to the last frame any camera captured; one that stopped
    # sooner holds its final frame for the tail, so frame i of every .mp4 is still the same
    # wall-clock instant.
    t_end = max(ts[-1] for ts in timestamp_lists)
    if t_end <= t_start:
        _log.warning("No overlapping SVO timestamp window; falling back to unsynced conversion")
        for svo_path, mp4_path in items:
            convert_svo_to_mp4(svo_path, mp4_path, crf=crf)
        return {}

    duration_ms = t_end - t_start
    n_frames = max(1, int(duration_ms * output_fps / 1000) + 1)
    sample_times = [t_start + int(i * 1000 / output_fps) for i in range(n_frames)]

    sync_meta = {
        "output_fps": output_fps,
        "t_start_ms": t_start,
        "t_end_ms": t_end,
        "n_frames": n_frames,
        "cameras": {},
    }

    for (svo_path, mp4_path), timestamps in zip(items, timestamp_lists):
        _log.info(
            "Syncing %s (%d SVO frames, offset %+d ms vs grid start)",
            svo_path.name,
            len(timestamps),
            timestamps[0] - t_start,
        )
        # Frames past this camera's own last timestamp are the held tail described above.
        n_real = sum(1 for t in sample_times if t <= timestamps[-1])
        ended_early_ms = t_end - timestamps[-1]
        sync_meta["cameras"][mp4_path.stem] = {
            "svo_path": svo_path.name,
            "first_ts_ms": timestamps[0],
            "last_ts_ms": timestamps[-1],
            "n_svo_frames": len(timestamps),
            "start_offset_ms": timestamps[0] - t_start,
            "n_real_frames": n_real,
            "held_tail_frames": n_frames - n_real,
            "ended_early_ms": ended_early_ms,
        }
        if ended_early_ms > EARLY_STOP_WARN_MS:
            # Loud, because the run itself carried on and looked fine: the operator only
            # finds out later that one angle stops mid-motion.
            _log.warning(
                "%s stopped recording %.1fs before the other camera(s) did -- %d of %d frames "
                "are real and the rest hold its last frame. Its grab loop died mid-run; check "
                "the run log for 'Error grabbing frame during recording'.",
                svo_path.name,
                ended_early_ms / 1000,
                n_real,
                n_frames,
            )
        convert_svo_to_mp4_resampled(
            svo_path,
            mp4_path,
            sample_times,
            timestamps,
            output_fps,
            crf=crf,
        )

    return sync_meta


def convert_svo_to_mp4(svo_path: Path, mp4_path: Path, crf: int = 20):
    """Convert SVO file to MP4 video (left RGB only) using ffmpeg for efficient compression."""
    import shutil
    import subprocess

    import pyzed.sl as sl

    if not svo_path.exists():
        raise FileNotFoundError(f"SVO file not found: {svo_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH, required for SVO to MP4 conversion")

    # Open SVO file
    _log.info(f"Converting {svo_path} to {mp4_path}")
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.svo_real_time_mode = False
    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO file: {err}")

    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height
    fps = zed.get_camera_information().camera_configuration.fps

    # Use ffmpeg with CRF for much smaller files with good quality
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Pipe frames from SVO to ffmpeg
    left_image = sl.Mat()
    rt_param = sl.RuntimeParameters()
    nb_frames = zed.get_svo_number_of_frames()
    try:
        with tqdm(total=nb_frames, desc="Converting SVO to MP4", unit="frame") as pbar:
            while True:
                err = zed.grab(rt_param)
                if err == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(left_image, sl.VIEW.LEFT)
                    frame_bgra = left_image.get_data()
                    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                    try:
                        proc.stdin.write(frame_bgr.tobytes())
                    except BrokenPipeError:
                        break  # ffmpeg died; returncode check below will raise
                    pbar.update(1)
                elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                else:
                    raise RuntimeError(f"Failed to grab frame at position {zed.get_svo_position()}: {err}")
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read()
        proc.wait()
        zed.close()

    if proc.returncode != 0:
        stderr_msg = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {stderr_msg}")
    _log.info(f"Conversion complete: {mp4_path}")
