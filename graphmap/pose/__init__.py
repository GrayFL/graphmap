"""Pose algebra; public imports stay at ``graphmap.pose``."""
from ._core import (
    Pose, PoseArray, PoseLike, Transform, TransformArray, Sim3, Sim3Array,
    Similarity3, Similarity3Array, parse_pose_label,
)

__all__ = [
    "Pose", "PoseArray", "PoseLike", "Transform", "TransformArray", "Sim3",
    "Sim3Array", "Similarity3", "Similarity3Array", "parse_pose_label",
]

# Keep pickle paths and introspection stable across the module-to-package move.
Pose.__module__ = __name__
PoseArray.__module__ = __name__
