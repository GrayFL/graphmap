"""dataset_wrapper samples -> graphmap observations; no dataset IO implementation."""
import json
import os
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from graphmap import Environment
from graphmap.adapters import rgbd_observation
from graphmap.pose import Pose
from graphmap.sensor import CameraIntrinsic


def observation_from_sample(sample, camera, timestamp=0.0):
    data, pose = sample["data"], sample["annotations"]["pose"]
    assert sample["metadata"]["coordinate_frames"]["export_pose"] == "floor_relative_rgb_camera"
    return rgbd_observation(
        data["depth"], intrinsic=camera,
        pose_camera=Pose(position=pose["position"], rotation=pose["quaternion_xyzw"]),
        rgb=data["rgb"], depth_scale=.001, stride=1, timestamp=timestamp,
        attributes={} if data["semantic"] is None else {"semantic_id": data["semantic"]},
    )


@pytest.mark.parametrize("reading", ["direct", "prefetch", "memory"])
def test_dataset_wrapper_hm3d_to_mapping(tmp_path, reading):
    HM3D = pytest.importorskip("dataset_wrapper.datasets.hm3d").HM3D
    Image = pytest.importorskip("PIL.Image")
    from scipy.spatial.transform import Rotation

    scene = tmp_path / "0000"
    scene.mkdir()
    rgb = np.full((3, 4, 3), [10, 20, 30], np.uint8)
    depth = np.full((3, 4), 2000, np.uint16)
    depth[0, 0] = 0
    labels = np.arange(12, dtype=np.uint32).reshape(3, 4)
    for frame in [0, 3]:
        Image.fromarray(rgb).save(scene / f"{frame:06d}_rgb.png")
        Image.fromarray(depth).save(scene / f"{frame:06d}_depth.png")
    # Second frame intentionally has no semantics; it must still be observed.
    np.savez(scene / "000000_semantic.npz", semantic=labels)
    (scene / "navigation_replay.json").write_text(json.dumps({"signature": {
        "coordinate_frames": {"export_pose": "floor_relative_rgb_camera"},
    }}))
    (scene / "runtime_log.pkl").write_bytes(pickle.dumps({
        frame: {"curr_pos": [1, 2 + frame, 0], "curr_rot": Rotation.from_euler("z", 90, degrees=True)}
        for frame in [0, 3]
    }))
    ds = HM3D(tmp_path, episodes=["0000"], load_replay=True)
    if reading == "direct":
        samples = list(ds)
    else:
        with ds.loader(num_workers=2, batch_size=None, prefetch=2) as loader:
            samples = list(loader)
        if reading == "memory":
            cache = {sample["key"]: sample for sample in samples}
            memory = ds.with_reader(lambda record: cache[record.key])
            # Prove graphmap consumes the cached data without reopening files.
            for path in scene.glob("*.png"):
                path.unlink()
            samples = list(memory)
    assert [sample["key"] for sample in samples] == [("0000", "000000"), ("0000", "000003")]
    camera = CameraIntrinsic.from_sensor(width=4, height=3, fx=2, fy=3, ppx=1, ppy=1)
    env = Environment(resolution=.01)
    for index, sample in enumerate(samples):
        observation = observation_from_sample(sample, camera, timestamp=index)
        env.observe(**observation)
        h, w = observation["pixel_indices"].T
        assert len(h) == 11
        assert_array_equal(observation["colors"], rgb[h, w])
        center = np.flatnonzero((h == 1) & (w == 1))[0]
        assert_allclose(observation["points_world"][center], [1, 4 + index * 3, 0])
    assert env.frame_count == 2 and env.N == 22
    assert env.index_db("semantic_id").num_links == 11
    snapshot = tmp_path / "map"
    env.save(snapshot)
    restored = Environment.load(snapshot)
    assert restored.get_index_values("semantic_id") == env.get_index_values("semantic_id")


@pytest.mark.dataset
def test_local_hm3d_rgbd_semantics_roundtrip(tmp_path):
    root = os.environ.get("GRAPHMAP_HM3D_ROOT")
    if not root:
        pytest.skip("set GRAPHMAP_HM3D_ROOT to enable the local dataset integration")
    from dataset_wrapper.datasets.hm3d import HM3D

    scene = os.environ.get("GRAPHMAP_HM3D_SCENE", "0000")
    ds = HM3D(root, episodes=[scene], frame_limit=3, load_replay=True)
    assert len(ds) == 3
    camera = CameraIntrinsic.from_sensor(width=640, height=640, hfov=np.deg2rad(64), vfov=np.deg2rad(64))
    first = ds.record(0)
    assert first.metadata["config_sha256"]["simulator_config"] == "7570bfa76cd07c69eee812f2b170ac0ef492a190c617e0eb897d797be7617bec"
    env = Environment()
    with ds.loader(num_workers=2, batch_size=None, prefetch=2) as loader:
        for index, sample in enumerate(loader):
            data = sample["data"]
            assert data["semantic"] is not None
            pose = sample["annotations"]["pose"]
            assert sample["metadata"]["coordinate_frames"]["export_pose"] == "floor_relative_rgb_camera"
            obs = rgbd_observation(
                data["depth"], intrinsic=camera,
                pose_camera=Pose(position=pose["position"], rotation=pose["quaternion_xyzw"]),
                rgb=data["rgb"], attributes={"semantic_id": data["semantic"]},
                depth_scale=.001, stride=32, timestamp=index * .1,
            )
            h, w = obs["pixel_indices"].T
            assert len(h) > 0
            assert_array_equal(obs["colors"], data["rgb"][h, w])
            assert obs["extra_keys"]["semantic_id"].get_batch(range(len(h))) == data["semantic"][h, w].tolist()
            env.observe(**obs)
    assert env.index_db("semantic_id").num_links == env.N
    env.save(tmp_path)
    restored = Environment.load(tmp_path)
    assert restored.get_index_values("semantic_id") == env.get_index_values("semantic_id")


@pytest.mark.dataset
def test_local_hm3d_ground_support_0000(tmp_path):
    """Measured support, a confirmed route, 3D properties and persisted 2D agree."""
    root = os.environ.get("GRAPHMAP_HM3D_ROOT")
    if not root:
        pytest.skip("set GRAPHMAP_HM3D_ROOT to enable real ground navigation")
    from dataset_wrapper.datasets.hm3d import HM3D
    from graphmap.traversability import GroundMap, GroundState

    ds = HM3D(root, episodes=["0000"], load_replay=True)
    assert len(ds) > 450
    assert ds.record(0).metadata["config_sha256"]["simulator_config"] == "7570bfa76cd07c69eee812f2b170ac0ef492a190c617e0eb897d797be7617bec"
    camera = CameraIntrinsic.from_sensor(width=640, height=640, hfov=np.deg2rad(64), vfov=np.deg2rad(64))
    selected = ds.iloc[np.unique(np.r_[np.linspace(0, len(ds) - 1, 96, dtype=int), 450])]
    env = Environment(resolution=.05)
    with selected.loader(num_workers=2, batch_size=None, prefetch=2) as samples:
        for order, sample in enumerate(samples):
            data, pose = sample["data"], sample["annotations"]["pose"]
            env.observe(**rgbd_observation(
                data["depth"], intrinsic=camera,
                pose_camera=Pose(pose["position"], pose["quaternion_xyzw"]),
                rgb=data["rgb"], depth_scale=.001, depth_range=(.15, 8), stride=8,
                timestamp=order * .1,
            ))
    reference = ds.iloc[450]  # This room has space for the specified 0.8 m square.
    frames = reference["metadata"]["coordinate_frames"]
    pose = reference["annotations"]["pose"]
    T_world_camera = Pose(pose["position"], pose["quaternion_xyzw"], wrd_frame="world", ego_frame="camera")
    T_body_camera = Pose(frames["camera_local_position"], frames["camera_local_quaternion_xyzw"],
                         wrd_frame="body", ego_frame="camera")
    T_world_body = T_world_camera * ~T_body_camera
    nav = GroundMap().update(env.infopoints.xyz, robot_pose=T_world_body, ground_z=T_world_body.position[2],
                             depth=reference["data"]["depth"], depth_scale=.001, depth_range=(.15, 8),
                             intrinsic=camera, pose_camera=T_world_camera)
    assert len(nav.roots) and np.count_nonzero(nav.confidence == 1) > 10
    start = nav.xyz[nav.roots[0]]
    confirmed = nav.xyz[nav.confidence == 1]
    goal = confirmed[np.argmax(np.linalg.norm(confirmed - start, axis=1))]
    path = nav.plan_path(start, goal)
    assert len(path) > 3
    attributes = nav.point_attributes(env.infopoints.xyz)
    assert np.count_nonzero(attributes["ground_confidence"] == 1) > 100
    assert np.count_nonzero(attributes["ground_state"] == GroundState.CONNECTED) > 100
    env.set_point_attributes(attributes)
    env.set_map2d(nav.to_map2d())
    env.save(tmp_path / "environment")
    nav.save(tmp_path / "ground.npz")
    restored = Environment.load(tmp_path / "environment")
    for name, values in attributes.items():
        assert_array_equal(restored.get_index_values(name), values)
    assert_array_equal(restored.map2d.grid, env.map2d.grid)
    assert_array_equal(GroundMap.load(tmp_path / "ground.npz").plan_path(start, goal), path)
