"""Domain boundaries used by optional robotics runtimes (no runtime imports)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from graphmap import Environment
from graphmap.adapters import pointcloud_observation, rgbd_observation
from graphmap.pose import Pose
from graphmap.sensor import CameraIntrinsic


def observe(environment, timestamp, *, x=1.0):
    environment.observe(**pointcloud_observation(
        [[x, 0, 0]], pose_sensor=Pose(), timestamp=timestamp,
    ))


def test_radian_rotation_and_xyzw_match_degree_pose():
    radian_pose = Pose(yaw=np.pi / 2, degrees=False)
    quaternion_pose = Pose(rotation=[0, 0, np.sqrt(.5), np.sqrt(.5)])
    for pose in (radian_pose, quaternion_pose, Pose(yaw=90)):
        assert_allclose(pose([1, 0, 0]), [0, 1, 0], atol=1e-12)


def test_millimeter_and_meter_depth_have_identical_world_geometry():
    camera = CameraIntrinsic.from_sensor(width=3, height=2, fx=4, fy=4, ppx=1, ppy=1)
    pose = Pose(x=1, yaw=np.pi / 2, degrees=False)
    millimeters = rgbd_observation(
        np.full((2, 3), 2000, dtype=np.uint16), intrinsic=camera,
        pose_camera=pose, depth_scale=.001,
    )
    meters = rgbd_observation(
        np.full((2, 3), 2.0), intrinsic=camera, pose_camera=pose,
    )
    assert_allclose(millimeters['points_world'], meters['points_world'])
    assert_allclose((~pose)(meters['points_world'])[:, 0], 2)


def test_native_nanosecond_origin_and_episode_are_callers_responsibility():
    # Subtract the integer origin before float conversion to retain precision.
    origin_ns = 1_793_275_200_000_000_000
    first = Environment(resolution=.1)
    observe(first, (origin_ns + 1_250_000_000 - origin_ns) / 1e9)
    assert_array_equal(first.infopoints.timestamp_ms, [1250])
    # A reset/seek creates a separate domain map; old timestamps cannot win.
    second = Environment(resolution=.1)
    observe(second, (origin_ns - origin_ns) / 1e9, x=1.01)
    assert_array_equal(second.infopoints.timestamp_ms, [0])
    assert_allclose(second.infopoints.xyz[:, 0], [1.01])
    assert_array_equal(first.infopoints.timestamp_ms, [1250])


def test_integer_nanosecond_origin_preserves_millisecond_truncation_boundaries():
    # Float conversion before subtraction loses sub-microsecond information at
    # Unix-sized origins, which can change the stored millisecond at a boundary.
    origin_ns = 1_793_275_200_000_000_003
    offsets_ns = np.array([999_999, 1_000_000, 1_000_001, 1_999_999, 2_000_001], dtype=np.int64)
    source_ns = origin_ns + offsets_ns
    relative_seconds = (source_ns - origin_ns) / 1e9
    environment = Environment(resolution=.1)
    environment.observe(**pointcloud_observation(
        [[float(index + 1), 0, 0] for index in range(len(offsets_ns))],
        pose_sensor=Pose(), timestamp=relative_seconds,
    ))
    assert_array_equal(environment.infopoints.timestamp_ms, [0, 1, 1, 1, 2])


def test_reset_map_session_does_not_reuse_point_semantics_or_mutable_snapshot():
    previous = Environment(resolution=.1)
    previous.observe(**pointcloud_observation(
        [[1., 0, 0]], pose_sensor=Pose(), timestamp=12.,
        attributes={'semantic_id': [17]},
    ))
    old_snapshot = previous.infopoints.copy()
    current = Environment(resolution=.1)
    current.observe(**pointcloud_observation(
        [[1.01, 0, 0]], pose_sensor=Pose(), timestamp=0.,
        attributes={'semantic_id': [23]},
    ))
    # Point keys describe geometry, not session identity. The upstream owner
    # must replace the Environment; graphmap must not infer a runtime epoch.
    assert_array_equal(previous.point_keys(), current.point_keys())
    assert previous.get_index_values('semantic_id') == [17]
    assert current.get_index_values('semantic_id') == [23]
    assert_array_equal(old_snapshot.timestamp_ms, [12_000])
    assert_array_equal(current.infopoints.timestamp_ms, [0])
    assert not np.shares_memory(old_snapshot.data, previous.infopoints.data)
    assert not np.shares_memory(old_snapshot.data, current.infopoints.data)
    current.infopoints.xyz[:] = 99
    assert_allclose(previous.infopoints.xyz, [[1., 0, 0]])
    assert_allclose(old_snapshot.xyz, [[1., 0, 0]])


@pytest.mark.parametrize('timestamp', [
    float('nan'), float('inf'), -float('inf'), 1_793_275_200,
    [float('nan')], [float('inf')], [1_793_275_200],
    [np.iinfo(np.int32).max / 1000 + 1],
    [np.iinfo(np.int32).min / 1000 - 1],
])
def test_observe_rejects_invalid_or_overflowing_relative_time(timestamp):
    environment = Environment()
    with pytest.raises(ValueError, match='relative seconds'):
        observe(environment, timestamp)
    assert environment.N == 0
    assert environment.frame_count == 0


def test_timestamp_limits_and_fractional_milliseconds_keep_existing_format():
    environment = Environment()
    points = [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
    environment.observe(**pointcloud_observation(
        points, pose_sensor=Pose(), timestamp=[
            np.iinfo(np.int32).min / 1000,
            np.iinfo(np.int32).max / 1000,
            1.2349,
        ],
    ))
    assert_array_equal(environment.infopoints.timestamp_ms,
                       [np.iinfo(np.int32).min, np.iinfo(np.int32).max, 1234])


def test_observation_and_explicit_map_copy_do_not_share_point_buffers():
    points = np.array([[1., 0, 0], [2., 0, 0]])
    environment = Environment()
    environment.observe(**pointcloud_observation(
        points, pose_sensor=Pose(), timestamp=0.,
    ))
    snapshot = environment.infopoints.copy()
    points[:] = 99
    observe(environment, 1., x=3.)
    assert_allclose(snapshot.xyz, [[1, 0, 0], [2, 0, 0]])
    assert environment.N == 3
    snapshot.xyz[:] = -99
    assert not np.any(environment.infopoints.xyz == -99)
