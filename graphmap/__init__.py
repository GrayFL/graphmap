"""
graphmap - 3D 点云语义地图构建工具库

提供基于 InfoPoints 的环境管理、编解码、可见性判定、2D 可行地图等功能。

核心组件:
    - InfoPoints: (N, 6) float32 紧凑存储的 3D 点云
    - IndexDB:    以 uint64 point key 为键的命名附加信息表
    - Environment: 环境管理器，维护全局地图并处理增量更新
    - Map2D:       2D 可行地图（遍历性栅格）
    - Pose:        6-DoF 位姿
    - encoding:    编解码工具
    - visibility:  可见性判定（视锥体裁剪、遮挡检测）
"""

# encoding
from .graphmap.encoding import (
    pack_rgba,
    unpack_rgba,
    pack_rgba_batch,
    unpack_rgba_batch,
    pack_timestamp,
    unpack_timestamp,
    pack_timestamp_batch,
    unpack_timestamp_batch,
    pack_index,
    unpack_index,
    pack_index_batch,
    unpack_index_batch,
    encode_4int8_to_1float32,
    decode_1float32_to_4int8,
    encode_4uint8_to_1float32,
    decode_1float32_to_4uint8,
    encode_int32_to_float32,
    decode_float32_to_int32,
    encode_uint32_to_float32,
    decode_float32_to_uint32,
    )

# pose
from .graphmap.pose import Pose

# infopoints
from .graphmap.infopoints import InfoPoints, COL_LAYOUT, NUM_COLS

# index_db
from .graphmap.index_db import IndexDB, point_keys_from_xyz, unpack_point_keys

# map2d
from .graphmap.map2d import Map2D
from .graphmap.topdown_map2d import (
    HABITAT_NAVIGATION_LAYER,
    HABITAT_COLOR_LAYER,
    ONLINE_OCCUPANCY_LAYER,
    TOPDOWN_NUM_LAYERS,
    TopdownMap2D,
)

# sensor
from .graphmap.sensor import (
    CameraIntrinsic,
    SensorBridge,
    load_observation_from_json,
    )

# visibility
from .graphmap.visibility import (
    compute_frustum_planes,
    are_points_in_frustum,
    frustum_culling,
    frustum_xy_bounds
    )

# environment
from .graphmap.environment import Environment

__all__ = [
    # encoding
    "pack_rgba",
    "unpack_rgba",
    "pack_rgba_batch",
    "unpack_rgba_batch",
    "pack_timestamp",
    "unpack_timestamp",
    "pack_timestamp_batch",
    "unpack_timestamp_batch",
    "pack_index",
    "unpack_index",
    "pack_index_batch",
    "unpack_index_batch",
    "encode_4int8_to_1float32",
    "decode_1float32_to_4int8",
    "encode_4uint8_to_1float32",
    "decode_1float32_to_4uint8",
    "encode_int32_to_float32",
    "decode_float32_to_int32",
    "encode_uint32_to_float32",
    "decode_float32_to_uint32",  # pose
    "Pose",  # infopoints
    "InfoPoints",
    "COL_LAYOUT",
    "NUM_COLS",  # index_db
    "IndexDB",
    "point_keys_from_xyz",
    "unpack_point_keys",  # map2d
    "Map2D",
    "TopdownMap2D",
    "HABITAT_NAVIGATION_LAYER",
    "HABITAT_COLOR_LAYER",
    "ONLINE_OCCUPANCY_LAYER",
    "TOPDOWN_NUM_LAYERS",  # sensor
    "CameraIntrinsic",
    "SensorBridge",
    "load_observation_from_json",
    "compute_frustum_planes",
    "are_points_in_frustum",
    "frustum_culling",
    "frustum_xy_bounds",
    "project_lidar_bins",
    "dilate_finite_max",
    "lidar_depth_radius_values",
    "lidar_depth_dilation_radii",
    "dilate_finite_max_adaptive",
    "lookup_finite_max_adaptive",  # environment
    "Environment",
    ]
