import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from graphmap import Environment
from graphmap.adapters import pointcloud_observation, rgbd_observation
from graphmap.misc.config import default_config
from graphmap.pose import Pose
from graphmap.sensor import CameraIntrinsic, SensorBridge, sample_hw_data


@pytest.fixture
def camera():
    return CameraIntrinsic.from_sensor(width=4, height=3, fx=2, fy=3, ppx=1, ppy=1)


def test_projection_matches_optical_pinhole_and_roundtrips(camera):
    graph = np.array([[2, -.5, -.2], [1, .2, .3]])
    hwd = camera.cam_to_hwd(graph)
    assert_allclose(hwd[:, 0], 1 - 3 * graph[:, 2] / graph[:, 0])
    assert_allclose(hwd[:, 1], 1 - 2 * graph[:, 1] / graph[:, 0])
    assert_allclose(hwd[:, 2], -graph[:, 0])
    assert_allclose(camera.hwd_to_cam(hwd), graph)
    assert_array_equal(camera.create_grid()[0], [0, 0])
    assert_array_equal(camera.create_grid(order="center")[0], [1, 1])


def test_intrinsic_fov_construction():
    camera = CameraIntrinsic.from_sensor(width=640, height=480, hfov=np.pi/2, vfov=np.pi/3)
    assert_allclose(camera.fx, 320)
    assert_allclose(camera.hfov, np.pi/2)
    assert camera.fw < 0 and camera.fh < 0


def test_rgbd_filters_all_aligned_data_and_maps_semantics(camera):
    depth = np.full((3, 4), 2000.0)
    depth[0] = [0, np.nan, np.inf, -1]
    mask = np.ones((3, 4), bool)
    mask[1, 3] = False
    labels = np.arange(12).reshape(3, 4)
    rgb = np.repeat(labels[..., None], 3, axis=-1).astype(np.uint8)
    pose = Pose(x=1, yaw=90)
    observation = rgbd_observation(depth, intrinsic=camera, pose_camera=pose, rgb=rgb,
                                   attributes={"semantic_id": labels}, mask=mask, depth_scale=.001)
    h, w = observation["pixel_indices"].T
    assert len(h) == 7
    assert_array_equal(observation["colors"], rgb[h, w])
    assert observation["extra_keys"]["semantic_id"].get_batch(range(7)) == labels[h, w].tolist()
    assert_allclose((~pose)(observation["points_world"])[:, 0], 2)
    assert_allclose(observation["points_hwd"][:, :2], observation["pixel_indices"])
    env = Environment(resolution=.01)
    env.observe(**observation)
    assert sorted(env.get_index_values("semantic_id")) == sorted(labels[h, w].tolist())


def test_range_depth_and_empty_input(camera):
    obs = rgbd_observation(np.full((3, 4), 2.), intrinsic=camera, pose_camera=Pose(), depth_kind="range")
    assert_allclose(np.linalg.norm(obs["points_world"], axis=1), 2)
    empty = rgbd_observation(np.zeros((3, 4)), intrinsic=camera, pose_camera=Pose())
    assert empty["points_world"].shape == (0, 3)
    env = Environment()
    env.observe(**empty)
    assert env.N == 0


@pytest.mark.parametrize("kwargs", [
    {"depth_scale": 0}, {"depth_kind": "guess"}, {"depth_range": (2, 1)},
    {"rgb": np.zeros((2, 2, 3))}, {"stride": (1.5, 2)},
])
def test_rgbd_rejects_ambiguous_or_misaligned_input(camera, kwargs):
    with pytest.raises(ValueError):
        rgbd_observation(np.ones((3, 4)), intrinsic=camera, pose_camera=Pose(), **kwargs)


def test_pointcloud_input_and_sampling(camera):
    with pytest.raises(ValueError):
        pointcloud_observation([[np.nan, 0, 0]], pose_sensor=Pose())
    with pytest.raises(ValueError):
        pointcloud_observation([[1, 0, 0]], pose_sensor=Pose(scale=2))
    values = np.arange(12).reshape(3, 4)
    result = sample_hw_data(values, [1., -1, np.nan], [1., 0, 0], default_value=-1)
    assert_array_equal(result.samples, [5, -1, -1])


def test_legacy_bridge_defaults_and_rsim_factory(camera):
    config = default_config()
    bridge = SensorBridge(config, camera)
    obs = {"tf": {"ret": [[1, 0, 0], [0, 0, 0, 1]], "timestamp": {"data": 1.0}},
           "camera": {"ret": np.full((3, 4, 4), 128, dtype=np.uint8)},
           "lidar3d": {"ret": [[2, 0, 0]]}}
    result = bridge.process_observation(obs)
    assert_allclose(result["points_world"], [[3, 0, 0]])
    assert_array_equal(result["colors"], [[128, 128, 128]])
    config.sensor_params.rgb_sensor = dict(width=4, height=3, fx=2, fy=3, ppx=1, ppy=1,
                                           position=[.5, 0, 1], rotation=[0, 0, 0])
    config.sensor_params.lidar3d_sensor = dict(position=[0, 0, 0], rotation=[0, 0, 0])
    built = SensorBridge.from_rsim_config(config)
    assert built.intrinsic.width == 4
    assert_allclose(built.ext_body2camera([0, 0, 0]), [.5, 0, 1])
