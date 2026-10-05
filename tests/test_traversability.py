import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from graphmap import Environment, GroundConfig, GroundMap, GroundState
from graphmap.adapters import pointcloud_observation
from graphmap.pose import Pose
from graphmap.sensor import CameraIntrinsic
from graphmap.traversability import estimate_surface_normals


def floor_cloud(height=lambda x, y: np.zeros_like(x), *, xlim=(-2.4, 2.4), ylim=(-1.6, 1.6)):
    x, y = np.meshgrid(np.arange(xlim[0], xlim[1] + .001, .05),
                       np.arange(ylim[0], ylim[1] + .001, .05), indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel(), height(x, y).ravel()])
    return points, np.tile([0., 0., 1.], (len(points), 1))


def row_at(nav, xyz):
    row = int(np.argmin(np.linalg.norm(nav.xyz - xyz, axis=1)))
    assert np.linalg.norm(nav.xyz[row] - xyz) < .06
    return row


def wall(x=.8):
    y, z = np.meshgrid(np.arange(-1.6, 1.61, .05), np.arange(.15, 1.01, .05), indexing="ij")
    return np.column_stack([np.full(y.size, x), y.ravel(), z.ravel()])


def test_flat_floor_connected_sparse_graph_and_raster_axes():
    pytest.importorskip("cv2")
    points, normals = floor_cloud()
    nav = GroundMap().update(points, normals=normals, robot_pose=Pose(), ground_z=0)
    assert nav.config.node_spacing == .4 and nav.config.support_spacing == .1
    assert len(nav.xyz) < len(points) / 8
    assert nav.confidence[row_at(nav, [0, 0, 0])] == 1
    path = nav.plan_path([0, 0, 0], [1.6, .8, 0])
    assert len(path) > 1
    assert_allclose(path[0], [0, 0, 0], atol=1e-8)
    assert_allclose(path[-1], [1.6, .8, 0], atol=1e-8)
    grid = nav.to_map2d()
    h, w = grid.world_to_grid([1.6, .8])
    assert grid.grid[h, w, 0] == 1
    assert grid.grid[h, w, 1] == 0
    attrs = nav.point_attributes(points, normals=normals)
    assert (attrs["ground_confidence"] == 1).sum() > 1000
    assert attrs["ground_candidate"].all()


def test_blind_prior_and_height_prior_do_not_override_obstacles():
    nav = GroundMap().update(np.empty((0, 3)), robot_pose=Pose(), ground_z=0)
    assert nav.confidence[row_at(nav, [0, 0, 0])] == pytest.approx(.6)
    assert np.all(nav.confidence < 1)
    assert len(nav.observation_targets()) > 0
    nav.update(np.array([[0., 0., .4]]), robot_pose=Pose(), ground_z=0)
    row = row_at(nav, [0, 0, 0])
    assert nav.confidence[row] == 0 and nav.state[row] == GroundState.OBSTACLE


def test_obstacle_inflation_and_disconnected_room():
    points, normals = floor_cloud()
    obstacle = wall()
    nav = GroundMap().update(np.vstack([points, obstacle]),
                             normals=np.vstack([normals, np.zeros_like(obstacle)]),
                             robot_pose=Pose(), ground_z=0)
    assert nav.confidence[row_at(nav, [.4, 0, 0])] == 0  # center is not on wall; footprint is
    far = row_at(nav, [1.6, 0, 0])
    assert nav.support[far] == 1 and 0 < nav.confidence[far] < 1
    assert nav.state[far] == GroundState.DISCONNECTED
    assert len(nav.plan_path([0, 0, 0], [1.6, 0, 0])) == 0
    attrs = nav.point_attributes(obstacle, normals=np.zeros_like(obstacle))
    assert not attrs["ground_candidate"].any()
    assert not attrs["ground_confidence"].any()


def test_pit_with_visible_bottom_blocks_and_path_detours():
    def height(x, y):
        return np.where((x >= .6) & (x <= 1.4) & (np.abs(y) <= .35), -.5, 0.)
    points, normals = floor_cloud(height)
    nav = GroundMap().update(points, normals=normals, robot_pose=Pose([-.8, 0, 0]), ground_z=0)
    edge = row_at(nav, [.4, 0, 0])
    assert nav.confidence[edge] == 0
    assert nav.state[edge] in {GroundState.DROP, GroundState.STEP, GroundState.OBSTACLE}
    path = nav.plan_path([-.8, 0, 0], [2., 0, 0])
    assert len(path) > 0
    assert np.max(np.abs(path[:, 1])) >= .8
    assert_allclose(path[:, 2], 0, atol=1e-8)


def test_missing_support_cannot_become_confirmed_by_nearness_or_dilation():
    points, normals = floor_cloud()
    keep = ~((np.abs(points[:, 0]) < .25) & (np.abs(points[:, 1]) < .25))
    nav = GroundMap().update(points[keep], normals=normals[keep], robot_pose=Pose(), ground_z=0)
    row = row_at(nav, [0, 0, 0])
    assert 0 < nav.confidence[row] < 1
    assert nav.support[row] < 1
    assert len(nav.plan_path([0, 0, 0], [1.6, 0, 0])) == 0


def test_ramp_connects_rooms_with_40cm_height_difference():
    def height(x, y):
        return np.clip((x + .4) / 2, 0, 1) * .4
    points, normals = floor_cloud(height, xlim=(-2.4, 3.2))
    ramp = (points[:, 0] > -.4) & (points[:, 0] < 1.6)
    normals[ramp] = [-.2, 0, 1]
    nav = GroundMap().update(points, normals=normals, robot_pose=Pose([-1.2, 0, 0]), ground_z=0)
    top = row_at(nav, [2.4, 0, .4])
    assert nav.confidence[top] == 1
    path = nav.plan_path([-1.2, 0, 0], [2.4, 0, .4])
    assert len(path) > 0 and path[:, 2].max() == pytest.approx(.4)
    # A high total elevation change is allowed, but an abrupt 40cm ledge isn't.
    points, normals = floor_cloud(lambda x, y: np.where(x >= .4, .4, 0.))
    nav.update(points, normals=normals, robot_pose=Pose([-.8, 0, 0]), ground_z=0)
    assert len(nav.plan_path([-.8, 0, 0], [1.6, 0, .4])) == 0


def test_stairs_and_excessive_slope_are_not_confirmed_floor():
    points, normals = floor_cloud(lambda x, y: np.clip(np.floor((x + .2) / .3), 0, 4) * .18)
    nav = GroundMap().update(points, normals=normals, robot_pose=Pose([-1.2, 0, 0]), ground_z=0)
    assert np.count_nonzero(nav.confidence == 0) > 0
    assert len(nav.plan_path([-1.2, 0, 0], [1.6, 0, .72])) == 0
    points, normals = floor_cloud(lambda x, y: .5 * x)
    normals[:] = [-.5, 0, 1]
    nav.update(points, normals=normals, robot_pose=Pose(), ground_z=0)
    row = row_at(nav, [0, 0, 0])
    assert nav.state[row] == GroundState.SLOPE and nav.confidence[row] == 0


def test_rectangular_footprint_respects_robot_yaw():
    points, normals = floor_cloud()
    x, z = np.meshgrid(np.arange(-2.4, 2.41, .05), np.arange(.2, .81, .1), indexing="ij")
    walls = np.vstack([np.column_stack([x.ravel(), np.full(x.size, y), z.ravel()]) for y in [-.3, .3]])
    points, normals = np.vstack([points, walls]), np.vstack([normals, np.zeros_like(walls)])
    nav = GroundMap(GroundConfig(footprint=(.8, .4))).update(points, normals=normals, robot_pose=Pose(), ground_z=0)
    assert nav.confidence[row_at(nav, [0, 0, 0])] == 1
    nav.update(points, normals=normals, robot_pose=Pose(rotation=[0, 0, 90]), ground_z=0)
    assert nav.confidence[row_at(nav, [0, 0, 0])] == 0


def test_swept_edge_blocks_obstacle_between_safe_endpoint_footprints():
    points, normals = floor_cloud()
    # This point misses both endpoints AND the segment midpoint; it intersects
    # the footprint at one quarter of the translation.
    obstacle = np.array([[.1, .53, .4]])
    nav = GroundMap().update(np.vstack([points, obstacle]),
                             normals=np.vstack([normals, np.zeros_like(obstacle)]),
                             robot_pose=Pose(rotation=[0, 0, 45]), ground_z=0)
    first, second = row_at(nav, [0, 0, 0]), row_at(nav, [.4, 0, 0])
    assert nav.confidence[first] == nav.confidence[second] == 1
    edge = np.flatnonzero(np.all(nav.edges == [first, second], axis=1))
    assert len(edge) == 1 and nav.edge_confidence[edge[0]] == 0
    assert len(nav.plan_path([0, 0, 0], [.4, 0, 0])) > 2


def test_free_snapped_node_does_not_validate_colliding_robot_stance():
    points, normals = floor_cloud()
    obstacle = np.array([[.58, 0., .4]])
    nav = GroundMap().update(np.vstack([points, obstacle]),
                             normals=np.vstack([normals, np.zeros_like(obstacle)]),
                             robot_pose=Pose([.2, 0, 0]), ground_z=0)
    assert nav.robot_confidence == 0
    assert not len(nav.roots) and not np.any(nav.confidence == 1)


def test_narrow_supported_strip_depends_on_robot_size():
    points, normals = floor_cloud(ylim=(-.3, .3))
    nav = GroundMap().update(points, normals=normals, robot_pose=Pose(), ground_z=0)
    assert nav.confidence[row_at(nav, [0, 0, 0])] < 1
    small = GroundMap(GroundConfig(footprint=(.4, .4))).update(points, normals=normals, robot_pose=Pose(), ground_z=0)
    assert small.confidence[row_at(small, [0, 0, 0])] == 1


def test_depth_proves_void_invalid_and_occluded_pixels_do_not(tmp_path):
    camera = CameraIntrinsic.from_sensor(width=64, height=64, fx=32, fy=32, ppx=32, ppy=32)
    pose = Pose([0, 0, 1], rotation=[0, 90, 0])  # X-forward camera looks down
    empty = np.empty((0, 3))
    kwargs = dict(robot_pose=Pose(), ground_z=0, intrinsic=camera, pose_camera=pose)
    nav = GroundMap()
    for value in [0., np.nan, .5]:
        nav.update(empty, depth=np.full((64, 64), value), **kwargs)
        assert nav.confidence[row_at(nav, [0, 0, 0])] > 0
    nav.update(empty, depth=np.full((64, 64), 2.), **kwargs)
    assert nav.state[row_at(nav, [0, 0, 0])] == GroundState.DROP
    nav.save(tmp_path / "ground.npz")
    nav = GroundMap.load(tmp_path / "ground.npz")
    nav.update(empty, depth=np.zeros((64, 64)), **kwargs)
    assert nav.confidence[row_at(nav, [0, 0, 0])] == 0  # unseen isn't cleared
    points, normals = floor_cloud()
    nav.update(points, normals=normals, depth=np.ones((64, 64)), **kwargs)
    assert nav.confidence[row_at(nav, [0, 0, 0])] == 1


def test_expected_support_inside_sensor_blind_range_is_not_negative_evidence():
    camera = CameraIntrinsic.from_sensor(width=64, height=64, fx=32, fy=32, ppx=32, ppy=32)
    nav = GroundMap().update(np.empty((0, 3)), robot_pose=Pose(), ground_z=0,
                             intrinsic=camera, pose_camera=Pose([0, 0, 1], rotation=[0, 90, 0]),
                             depth=np.full((64, 64), 2.), depth_range=(1.5, 3.))
    assert nav.robot_confidence == pytest.approx(.6)


def test_robot_height_contact_and_low_step_under_overhang():
    points, normals = floor_cloud()
    ceiling = np.array([[0, 0, 1.]])
    nav = GroundMap().update(np.vstack([points, ceiling]),
                             normals=np.vstack([normals, np.zeros_like(ceiling)]), robot_pose=Pose(), ground_z=0)
    assert nav.robot_confidence == 0
    # A low bump fits max_step, but its raised support must not erase an overhead collision.
    bump = np.array([[.2, 0, .1], [.25, 0, .1], [.3, 0, .1]])
    ceiling[0] = [.2, 0, 1.05]
    nav.update(np.vstack([points, bump, ceiling]),
               normals=np.vstack([normals, np.tile([0, 0, 1], (3, 1)), np.zeros_like(ceiling)]),
               robot_pose=Pose(), ground_z=0)
    # Both ground levels are observed; overhead clearance is judged conservatively.
    assert nav.robot_confidence == 0


def test_normals_separate_floor_and_wall_without_semantic_labels():
    points, _ = floor_cloud(xlim=(-1, 1), ylim=(-1, 1))
    vertical = wall(x=2)
    normals, roughness = estimate_surface_normals(np.vstack([points, vertical]))
    assert np.all(normals[:len(points), 2] > .99)
    assert np.max(np.abs(normals[len(points):, 2])) < .01
    assert np.max(roughness) < 1e-6
    nav = GroundMap().update(points, robot_pose=Pose(), ground_z=0)
    assert nav.confidence[row_at(nav, [0, 0, 0])] == 1


def test_recompute_old_point_attributes_preserves_semantics_and_roundtrip(tmp_path):
    pytest.importorskip("cv2")
    points, _ = floor_cloud()
    env = Environment(resolution=.025)
    env.observe(**pointcloud_observation(points, pose_sensor=Pose(), attributes={"semantic_id": np.full(len(points), 7)}))
    nav = GroundMap().update(env.infopoints.xyz, robot_pose=Pose(), ground_z=0)
    env.set_point_attributes(nav.point_attributes(env.infopoints.xyz))
    old_key = env.point_keys(np.array([[0., 0., 0.]]))[0]
    assert env.index_db("ground_confidence")[int(old_key)] == 1
    env.observe(**pointcloud_observation([[0, 0, .4]], pose_sensor=Pose(), timestamp=1))
    nav.update(env.infopoints.xyz, robot_pose=Pose(), ground_z=0)
    env.set_point_attributes(nav.point_attributes(env.infopoints.xyz))
    assert env.index_db("ground_confidence")[int(old_key)] == 0
    assert env.index_db("semantic_id")[int(old_key)] == 7
    env.set_map2d(nav.to_map2d())
    env.save(tmp_path / "environment")
    nav.save(tmp_path / "navigation.npz")
    restored = Environment.load(tmp_path / "environment")
    nav_loaded = GroundMap.load(tmp_path / "navigation.npz")
    for name in ["ground_confidence", "ground_state", "ground_candidate", "semantic_id"]:
        assert restored.get_index_values(name) == env.get_index_values(name)
    assert_array_equal(nav_loaded.to_map2d().grid, env.map2d.grid)
    assert_array_equal(restored.map2d.grid, env.map2d.grid)
    with pytest.raises(ValueError, match="align"):
        env.set_point_attributes({"ground_confidence": [1.]})
    assert env.index_db("ground_confidence")[int(old_key)] == 0
    derived_grid = restored.map2d.grid.copy()
    restored.observe(**pointcloud_observation([[1.2, .8, .4]], pose_sensor=Pose(), timestamp=2))
    assert_array_equal(restored.map2d.grid, derived_grid)


@pytest.mark.parametrize("kwargs", [{"footprint": (0, 1)}, {"node_spacing": .8},
                                   {"near_confidence": 1}, {"max_step": 2}, {"normal_neighbors": 2}])
def test_configuration_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        GroundConfig(**kwargs)
