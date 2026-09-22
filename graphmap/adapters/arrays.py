"""Array adapters shared by file readers, sensor drivers, and simulators.

No driver SDK, file access, or map mutation belongs here. Returned dictionaries
can be passed directly to ``Environment.observe(**observation)``.
"""
from collections.abc import Mapping

import numpy as np

from ..index_db import IndexDB
from ..pose import Pose
from ..sensor import CameraIntrinsic


def _attributes_to_tables(attributes, count):
    tables = {}
    for name, values in (attributes or {}).items():
        if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("attribute names must be nonempty file-safe table names")
        if len(values) != count:
            raise ValueError(f"attributes[{name!r}] must have {count} rows")
        table = IndexDB(name)
        for row, value in enumerate(values):
            # Convert NumPy scalars to portable Python values; arrays stay arrays.
            table.put(row, value.item() if isinstance(value, np.generic) else value)
        tables[name] = table
    return tables


def pointcloud_observation(
    points,
    *,
    pose_sensor: Pose,
    pose_robot: Pose | None = None,
    colors=None,
    attributes: Mapping | None = None,
    timestamp: float | None = None,
):
    """Normalize graphmap sensor-local points (N, 3), measured in metres.

    ``pose_sensor`` maps sensor -> graphmap world. Use Pose() for world points.
    Colors are uint8 RGB/RGBA (N, 3/4); channel order is never guessed.
    Attribute rows match point rows and become temporary IndexDB row keys.
    Timestamp is elapsed seconds from a caller-chosen session origin, NOT Unix
    epoch seconds. Invalid XYZ raises; filter all aligned arrays together first.
    """
    if not isinstance(pose_sensor, Pose) or not pose_sensor.is_rigid:
        raise ValueError("pose_sensor must be a rigid Pose in metric graphmap coordinates")
    if pose_robot is not None and (not isinstance(pose_robot, Pose) or not pose_robot.is_rigid):
        raise ValueError("pose_robot must be a rigid Pose")
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite (N, 3) array")
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape not in {(len(points), 3), (len(points), 4)} or colors.dtype != np.uint8:
            raise ValueError("colors must be uint8 RGB/RGBA with one row per point")
    return {
        "pose_robot": pose_sensor if pose_robot is None else pose_robot,
        # Historical observe parameter name; for RGB-D this is the camera pose.
        "pose_lidar3d": pose_sensor,
        "points_world": pose_sensor(points).astype(np.float32),
        "points_floor": None,
        "colors": colors,
        "timestamp": timestamp,
        "extra_keys": _attributes_to_tables(attributes, len(points)),
    }


def rgbd_observation(
    depth,
    *,
    intrinsic: CameraIntrinsic,
    pose_camera: Pose,
    rgb=None,
    attributes: Mapping | None = None,
    pose_robot: Pose | None = None,
    timestamp: float | None = None,
    depth_scale: float = 1.0,
    depth_kind: str = "axial",
    depth_range: tuple[float, float] = (0.0, np.inf),
    stride: int | tuple[int, int] = 1,
    mask=None,
):
    """Back-project aligned RGB/depth/attributes into a normalized observation.

    depth_scale multiplies stored depth to metres (uint16 mm -> 0.001).
    axial means forward-axis depth X; range means Euclidean ray distance.
    Zero/negative/nonfinite depth and masked pixels are removed from ALL data.
    depth_range is in metres of the chosen depth_kind, with exclusive bounds.
    RGB and each attribute image must match (H, W) of the undistorted depth.
    ``pixel_indices`` in the result maps each surviving row back to [h, w].
    """
    depth = np.asarray(depth)
    shape = (intrinsic.height, intrinsic.width)
    if depth.shape != shape:
        raise ValueError(f"depth must have shape {shape}")
    if depth.dtype.kind not in "fiu":
        raise ValueError("depth must contain real numeric values")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be finite and positive")
    if depth_kind not in {"axial", "range"}:
        raise ValueError("depth_kind must be 'axial' or 'range'")
    near, far = depth_range
    if not np.isfinite(near) or near < 0 or np.isnan(far) or far <= near:
        raise ValueError("depth_range must satisfy 0 <= near < far")
    hw = intrinsic.create_grid(stride=stride)
    h, w = hw.T
    distance = depth[h, w].astype(np.float64) * depth_scale
    valid = np.isfinite(distance) & (distance > near) & (distance < far)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError("mask must match depth shape")
        valid &= mask[h, w]
    hw, distance = hw[valid], distance[valid]
    h, w = hw.T
    rays = intrinsic.hwd_to_cam(np.column_stack([hw, -np.ones(len(hw))]))
    if depth_kind == "range":
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    points = rays * distance[:, None]

    def sample_aligned(data, name):
        data = np.asarray(data)
        if data.shape[:2] != shape:
            raise ValueError(f"{name} must match depth height and width")
        return data[h, w]

    result = pointcloud_observation(
        points, pose_sensor=pose_camera, pose_robot=pose_robot,
        colors=None if rgb is None else sample_aligned(rgb, "rgb"),
        attributes={name: sample_aligned(data, name) for name, data in (attributes or {}).items()},
        timestamp=timestamp,
    )
    result.update(
        pose_camera=pose_camera,
        pixel_indices=hw,
        points_hwd=intrinsic.cam_to_hwd(points),
        visible_mask=np.ones(len(points), dtype=bool),
    )
    return result
