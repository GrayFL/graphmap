"""Stable sensor API with on-demand imports for independent components."""
from importlib import import_module
from .camera import CameraIntrinsic

_EXPORTS = {
    "SensorBridge": ".bridge",
    "RayDirLUTEstimator": ".lidar",
    "sample_hw_data": ".sampling",
    "load_observation_from_json": ".io",
}
__all__ = ["CameraIntrinsic", *_EXPORTS]
CameraIntrinsic.__module__ = __name__


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
