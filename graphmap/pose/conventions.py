"""Explicit axis changes; these do not infer origins, units, or pose direction."""
from typing import Literal

import numpy as np

from ._core import Pose, PoseArray

Convention = Literal["graphmap", "opencv", "opengl", "habitat"]


def basis_to_graphmap(convention: Convention) -> np.ndarray:
    """Return C such that ``p_graph = C @ p_source`` (column vectors).

    OpenCV: right/down/forward -> forward/left/up.
    OpenGL/Habitat: right/up/backward -> forward/left/up.
    Habitat world conversion uses the same axes as TopdownMap2D; it does
    not subtract a floor height. Returned arrays can be modified safely.
    """
    if convention == "graphmap":
        return np.eye(3)
    if convention == "opencv":
        return np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    if convention in {"opengl", "habitat"}:
        return np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    raise ValueError(f"Unknown coordinate convention: {convention!r}")


def convert_points(points, *, source: Convention, target: Convention = "graphmap"):
    """Change axes of an array (..., 3), preserving shape and metric units."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    change = basis_to_graphmap(target).T @ basis_to_graphmap(source)
    return points @ change.T


def convert_pose(pose: Pose | PoseArray, *, world: Convention, local: Convention):
    """Convert BOTH bases of a local-to-world pose to graphmap.

    ``T_graph = C_world @ T_source @ inv(C_local)``. For an OpenCV
    camera pose in an already graphmap world, use world="graphmap",
    local="opencv". Invert a world-to-camera extrinsic before calling.
    Frame labels and scale are retained; caller owns the frame names.
    """
    left, right = np.eye(4), np.eye(4)
    left[:3, :3] = basis_to_graphmap(world)
    right[:3, :3] = basis_to_graphmap(local).T
    return type(pose).from_matrix(
        left @ pose.matrix @ right,
        wrd_frame=pose.wrd_frame, ego_frame=pose.ego_frame,
    )
