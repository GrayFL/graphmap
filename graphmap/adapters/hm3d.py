"""Reader for the local HM3D RGB-D export layout, not the upstream HM3D SDK.

Reads ``{frame:06d}_rgb.png``, ``_depth.png``, and optional ``_semantic.npz``.
Calibration, depth scale, pose direction and scene origin remain explicit
arguments to rgbd_observation; they cannot be inferred from these images.
"""
from pathlib import Path

import numpy as np


def load_hm3d_frame(scene_dir: str | Path, frame_id: int) -> dict:
    """Return rgb (RGB/RGBA), raw depth, and optional semantic label array.

    OpenCV is imported only when this reader is used. Missing RGB/depth files
    raise FileNotFoundError. Missing semantics are represented by None.
    """
    import cv2

    if not isinstance(frame_id, (int, np.integer)) or frame_id < 0:
        raise ValueError("frame_id must be a nonnegative integer")
    scene = Path(scene_dir)
    prefix = f"{frame_id:06d}"

    def read_image(suffix):
        path = scene / f"{prefix}_{suffix}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        return image

    color = read_image("rgb")
    if color.ndim != 3 or color.shape[2] not in (3, 4):
        raise ValueError("RGB image must have 3 or 4 channels")
    color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGBA if color.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    depth = read_image("depth")
    if depth.ndim != 2 or color.shape[:2] != depth.shape:
        raise ValueError("RGB and depth must have matching height and width")
    semantic_path = scene / f"{prefix}_semantic.npz"
    semantic = None
    if semantic_path.is_file():
        with np.load(semantic_path, allow_pickle=False) as data:
            semantic = data["semantic"]
        if semantic.shape != depth.shape:
            raise ValueError("semantic labels must match depth shape")
    return {"rgb": color, "depth": depth, "semantic": semantic}
