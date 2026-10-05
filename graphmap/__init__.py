"""Point-cloud geometry and mapping, with optional components imported on demand."""
from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    'pack_rgba': '.encoding',
    'unpack_rgba': '.encoding',
    'pack_rgba_batch': '.encoding',
    'unpack_rgba_batch': '.encoding',
    'pack_timestamp': '.encoding',
    'unpack_timestamp': '.encoding',
    'pack_timestamp_batch': '.encoding',
    'unpack_timestamp_batch': '.encoding',
    'pack_index': '.encoding',
    'unpack_index': '.encoding',
    'pack_index_batch': '.encoding',
    'unpack_index_batch': '.encoding',
    'encode_4int8_to_1float32': '.encoding',
    'decode_1float32_to_4int8': '.encoding',
    'encode_4uint8_to_1float32': '.encoding',
    'decode_1float32_to_4uint8': '.encoding',
    'encode_int32_to_float32': '.encoding',
    'decode_float32_to_int32': '.encoding',
    'encode_uint32_to_float32': '.encoding',
    'decode_float32_to_uint32': '.encoding',
    'Pose': '.pose',
    'InfoPoints': '.infopoints',
    'COL_LAYOUT': '.infopoints',
    'NUM_COLS': '.infopoints',
    'IndexDB': '.index_db',
    'point_keys_from_xyz': '.index_db',
    'unpack_point_keys': '.index_db',
    'Map2D': '.map2d',
    'GroundConfig': '.traversability',
    'GroundMap': '.traversability',
    'GroundState': '.traversability',
    'HABITAT_NAVIGATION_LAYER': '.topdown_map2d',
    'HABITAT_COLOR_LAYER': '.topdown_map2d',
    'ONLINE_OCCUPANCY_LAYER': '.topdown_map2d',
    'TOPDOWN_NUM_LAYERS': '.topdown_map2d',
    'TopdownMap2D': '.topdown_map2d',
    'CameraIntrinsic': '.sensor',
    'SensorBridge': '.sensor',
    'load_observation_from_json': '.sensor',
    'compute_frustum_planes': '.visibility',
    'are_points_in_frustum': '.visibility',
    'frustum_culling': '.visibility',
    'frustum_xy_bounds': '.visibility',
    'Environment': '.environment',
    'PoseArray': '.pose',
    'PoseLike': '.pose',
    'Transform': '.pose',
    'TransformArray': '.pose',
    'Sim3': '.pose',
    'Sim3Array': '.pose',
    'Similarity3': '.pose',
    'Similarity3Array': '.pose',
    'parse_pose_label': '.pose',
    'RayDirLUTEstimator': '.sensor',
    'sample_hw_data': '.sensor',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_EXPORTS[name], __name__), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
