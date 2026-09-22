import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose

from graphmap.pose import Pose, PoseArray, Transform, Sim3
from graphmap.pose.conventions import basis_to_graphmap, convert_points, convert_pose


def test_composition_inverse_and_frame_direction():
    body = Pose(x=1, yaw=90, wrd_frame="world", ego_frame="body")
    camera = Pose(x=0.5, z=1, wrd_frame="body", ego_frame="camera")
    world_camera = body * camera
    points = np.array([[2, 0, 0], [1, -1, 2]])
    assert_allclose(world_camera(points), body(camera(points)))
    assert_allclose(world_camera(points)[0], [1, 2.5, 1], atol=1e-12)
    assert_allclose((~world_camera)(world_camera(points)), points, atol=1e-12)
    assert world_camera.wrd_frame == "world"
    assert world_camera.ego_frame == "camera"
    with pytest.raises(ValueError):
        camera * body


def test_similarity_about_point_and_matrix_roundtrip():
    transform = Pose.from_about(rotation=[0, 0, 90], scale=2, about=[1, 2, 0])
    assert_allclose(transform([1, 2, 0]), [1, 2, 0])
    assert_allclose(transform([2, 2, 0]), [1, 4, 0])
    assert Pose.from_matrix(transform.matrix).allclose(transform)
    assert Transform is Sim3 is Pose
    assert pickle.loads(pickle.dumps(transform)).allclose(transform)
    assert Pose.__module__ == "graphmap.pose"


def test_nd_posearray_and_pointcloud_broadcasting():
    poses = PoseArray(position=np.zeros((2, 3, 3)), rotation=[0, 0, 90])
    poses.translation[..., 0] = np.arange(6).reshape(2, 3)
    points = np.array([[1, 0, 0], [2, 0, 0.]])
    result = poses.apply_all(points)
    assert result.shape == (2, 3, 2, 3)
    for i in range(2):
        for j in range(3):
            assert_allclose(result[i, j], poses[i, j](points))
    assert_allclose(poses.apply_clouds(result, inverse=True), np.broadcast_to(points, result.shape), atol=1e-12)
    assert PoseArray.from_matrix(poses.matrix).allclose(poses)


@pytest.mark.parametrize("convention", ["graphmap", "opencv", "opengl", "habitat"])
def test_basis_is_right_handed_and_roundtrips(convention):
    basis = basis_to_graphmap(convention)
    assert_allclose(basis @ basis.T, np.eye(3))
    assert_allclose(np.linalg.det(basis), 1)
    points = np.array([[1, 2, 3], [-2, 4, 1]])
    graph = convert_points(points, source=convention)
    assert_allclose(convert_points(graph, source="graphmap", target=convention), points)


def test_change_world_and_camera_basis_independently():
    optical = Pose(position=[1, 2, 3], rotation=[10, 20, 30])
    point_cv = np.array([0.2, 0.3, 2.0])
    for world in ["graphmap", "habitat"]:
        converted = convert_pose(optical, world=world, local="opencv")
        assert_allclose(
            converted(convert_points(point_cv, source="opencv")),
            convert_points(optical(point_cv), source=world),
        )
    assert_allclose(convert_points(point_cv, source="opencv"), [2, -0.2, -0.3])


@pytest.mark.parametrize("matrix", [np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1])])
def test_matrix_rejects_anisotropic_scale_and_reflection(matrix):
    with pytest.raises(ValueError):
        Pose.from_matrix(matrix)
