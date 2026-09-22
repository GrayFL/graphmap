"""Normalize external arrays at the boundary, then call Environment.observe."""
from .arrays import pointcloud_observation, rgbd_observation

__all__ = ["pointcloud_observation", "rgbd_observation"]
