import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from graphmap import Environment, InfoPoints, IndexDB, point_keys_from_xyz, unpack_point_keys
from graphmap.adapters import pointcloud_observation
from graphmap.pose import Pose


def test_point_keys_negative_boundary_and_legacy_bits():
    xyz = np.array([[-32768, 0, 32767], [-1, 2, 3.]]) * .05
    keys = point_keys_from_xyz(xyz, .05, np.array([0, 0xABCD1234], np.uint32))
    coordinates, legacy = unpack_point_keys(keys)
    assert_array_equal(coordinates, [[-32768, 0, 32767], [-1, 2, 3]])
    assert_array_equal(legacy, [0, 0xABCD])


@pytest.mark.parametrize("xyz,resolution", [([[np.nan, 0, 0]], .1), ([[0, 0, 0]], np.nan),
                                          ([[32768, 0, 0]], 1), ([[1e30, 0, 0]], 1)])
def test_point_key_rejects_invalid_geometry(xyz, resolution):
    with pytest.raises(ValueError):
        point_keys_from_xyz(xyz, resolution)


def test_bitpacked_points_indexing_and_serialization(tmp_path):
    points = InfoPoints.from_arrays([[1, 2, 3], [4, 5, 6]], [[255, 255, 255], [1, 2, 3]],
                                   [-1, 123], [0xFFFFFFFF, 1])
    points.metadata["frame"] = "world"
    assert points[0].shape == (1, 6)
    assert_array_equal(points[0].color, [[255, 255, 255, 255]])
    path = tmp_path / "points.npz"
    points.save(path)
    restored = InfoPoints.load(path)
    # Packed color/index float slots may be NaNs: compare bits, not float values.
    assert_array_equal(restored.data.view(np.uint32), points.data.view(np.uint32))
    assert restored.metadata == {"frame": "world"}


def test_index_db_shared_values_gc_and_roundtrip(tmp_path):
    db = IndexDB("semantic_id")
    a = db.put(1, {"class": "chair"})
    assert db.put(2, {"class": "chair"}) == a
    assert db.size == 1 and db.num_links == 2
    db.put(1, {"class": "table"})
    db.gc(np.array([1], dtype=np.uint64))
    assert db.size == db.num_links == 1
    db.save(tmp_path / "semantic.pkl")
    restored = IndexDB.load(tmp_path / "semantic.pkl")
    assert restored[1] == {"class": "table"}


def test_environment_newest_voxel_keeps_corresponding_semantics(tmp_path):
    env = Environment(resolution=.1)
    env.observe(**pointcloud_observation([[1, 0, 0], [1.01, 0, 0], [2, 0, 0]], pose_sensor=Pose(),
                                         attributes={"semantic_id": [1, 2, 3]}, timestamp=1.0))
    assert env.N == 2
    assert env.get_index_values("semantic_id", np.array([[1., 0, 0]])) == [2]
    # An older observation must not overwrite the winner's side table.
    env.observe(**pointcloud_observation([[1, 0, 0]], pose_sensor=Pose(),
                                         attributes={"semantic_id": [9]}, timestamp=.5))
    assert env.get_index_values("semantic_id", np.array([[1., 0, 0]])) == [2]
    env.save(tmp_path)
    restored = Environment.load(tmp_path)
    assert_array_equal(restored.infopoints.data.view(np.uint32), env.infopoints.data.view(np.uint32))
    assert restored.get_index_values("semantic_id") == env.get_index_values("semantic_id")
    assert restored.frame_count == 2


def test_rectangular_grid_xy_axes_slices_and_roundtrip(tmp_path):
    pytest.importorskip("cv2")
    from graphmap.map2d import Map2D
    grid = Map2D(resolution=.5, shape=(7, 11), origin=(2, 4))
    assert_array_equal(grid.world_to_grid([1, -1]), [4, 2])
    assert_allclose(grid.grid_to_world([4, 2]), [1, -1])
    grid[1., -1., 0] = 0
    assert grid.grid[4, 2, 0] == 0
    assert_array_equal(grid[:, :, 0], grid.grid[:, :, 0])
    grid.save(tmp_path / "grid.npz")
    restored = Map2D.load(tmp_path / "grid.npz")
    assert_array_equal(restored.grid, grid.grid)
    assert_allclose(restored.origin, grid.origin)


def test_topdown_habitat_grid_and_world_agree():
    pytest.importorskip("cv2")
    from graphmap.topdown_map2d import TopdownMap2D
    from graphmap.pose.conventions import convert_points
    navigation = np.arange(6).reshape(2, 3)
    color = np.zeros((2, 3, 3), np.uint8)
    bounds = np.array([[0, 0, 0], [3, 2, 2]])
    grid = TopdownMap2D.from_habitat_maps(navigation, color, bounds, 1, 0)
    assert_array_equal(grid.navigation_grid(), (navigation[::-1, ::-1] >= 1))
    corners = convert_points(bounds, source="habitat")[:, :2]
    assert_allclose(grid.graph_bounds, np.sort(corners, axis=0))


def test_default_environment_2d_update_load_and_continue(tmp_path):
    pytest.importorskip("cv2")
    env = Environment(resolution=.5, enable_2d_map=True, side_length=10)
    observation = pointcloud_observation([[1, 0, 0], [2, 0, 1]], pose_sensor=Pose())
    env.observe(**observation)
    assert env.map2d[1., 0., 0] == 0
    assert env.map2d[2., 0., 0] == 0
    env.save(tmp_path)
    restored = Environment.load(tmp_path)
    restored.observe(**pointcloud_observation([[3, 0, 1]], pose_sensor=Pose()))
    assert restored.map2d[3., 0., 0] == 0


def test_legacy_occupancy_height_filter(tmp_path):
    pytest.importorskip("cv2")
    from graphmap.map2d import Map2D
    from graphmap.misc.config import default_config, Munch
    config = default_config(side_length=10)
    config.agent = Munch(collision_height_min=.2, collision_height_max=1.5)
    grid = Map2D(config=config)
    grid.update_from_occupancy(np.array([[1, 0, 0], [2, 0, 1], [3, 0, 2]]))
    assert grid[1., 0., 0] == -1
    assert grid[2., 0., 0] == 0
    assert grid[3., 0., 0] == -1
    env = Environment(config, enable_2d_map=True)
    env.save(tmp_path)
    restored = Environment.load(tmp_path, config=config)
    restored.observe(**pointcloud_observation([[1, 0, 0], [2, 0, 1]], pose_sensor=Pose()))
    assert restored.map2d[1., 0., 0] == -1
    assert restored.map2d[2., 0., 0] == 0
