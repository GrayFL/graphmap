"""
可见性判定 — 基于视锥体裁剪判断已知点在当前视角下是否可见

支持：
    1. 视锥体裁剪 (frustum culling)
    2. 视距范围过滤
    3. 深度遮挡检测 (简化版)
"""

import numpy as np
from typing import Tuple, Optional

from .pose import Pose

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .sensor import CameraIntrinsic

from scipy import ndimage
from scipy.spatial import cKDTree

import logging

logger = logging.getLogger(__name__)


def _to_fov_half_pair(value, default: tuple = (45.0, 45.0)):
    """
    将 FOV 参数规范化为半角（度）。

    标量时对称分配；为 (a, b) 时直接使用（不对称 FOV）。
    例如 fov_h=(20, 80) 表示左侧 20°、右侧 80°，总 HFOV=100°。
    """
    if value is None:
        return tuple(default)
    if isinstance(value, (int, float)):
        half = float(value) / 2.0
        return (half, half)
    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            return (float(value[0]), float(value[1]))
        raise ValueError(
            f"FOV tuple must have exactly 2 elements, got {len(value)}"
            )
    raise TypeError(f"Unsupported FOV type: {type(value)}")


def compute_frustum_planes(
        camera_pose: Pose,
        fov_h=None,
        fov_v=None,
        near: float = 0.1,
        far: float = 100.0
    ) -> np.ndarray:
    """
    计算视锥体的 6 个平面方程。

    每个平面定义为 ax + by + cz + d = 0，法向量指向内部。

    相机局部坐标系: X=forward（前进方向）, Y=left（前进方向左边）, Z=up（上方）。

    Parameters
    ----------
    camera_pose : Pose
        相机在世界坐标系下的位姿
    fov_h : float or (float, float), optional
        水平视场角。标量时对称分配；tuple 时 (left_deg, right_deg)。
        默认 (45.0, 45.0) 即总 HFOV=90°。
    fov_v : float or (float, float), optional
        垂直视场角。标量时对称分配；tuple 时 (up_deg, down_deg)。
        默认 (30.0, 30.0) 即总 VFOV=60°。
    near : float
        近裁剪面距离（米）
    far : float
        远裁剪面距离（米）

    Returns
    -------
    np.ndarray
        shape (6, 4), 每行 [a, b, c, d]
    """
    half_left, half_right = _to_fov_half_pair(fov_h, (45.0, 45.0))
    half_up, half_down = _to_fov_half_pair(fov_v, (30.0, 30.0))

    half_left_rad = np.deg2rad(half_left)
    half_right_rad = np.deg2rad(half_right)
    half_up_rad = np.deg2rad(half_up)
    half_down_rad = np.deg2rad(half_down)

    # 相机局部坐标系: X=forward（前进方向）, Y=left（前进方向左边）, Z=up（上方）
    R = camera_pose.rotation.as_matrix()
    T = camera_pose.position.reshape(3, 1)

    forward = R @ np.array([1, 0, 0])  # X 轴 = 前进方向
    left = R @ np.array([0, 1, 0])  # Y 轴 = 左方向
    up = R @ np.array([0, 0, 1])  # Z 轴 = 上方向

    near_center = T.ravel() + forward*near
    far_center = T.ravel() + forward*far

    planes = np.empty((6, 4), dtype=np.float32)

    # 近平面: 法向量 = +forward (指向内部, 即远离相机), 过 near_center
    n_near = forward
    planes[0, :3] = n_near
    planes[0, 3] = -np.dot(n_near, near_center)

    # 远平面: 法向量 = -forward (指向内部, 即朝向相机), 过 far_center
    n_far = -forward
    planes[1, :3] = n_far
    planes[1, 3] = -np.dot(n_far, far_center)

    # 左平面: 法向量 ⊥ 面, 指向内部（向右）, 过相机原点
    # 平面包含 left 边界射线 (cos(half_left), sin(half_left), 0)_local 与 up 轴
    n_left = np.sin(half_left_rad) * forward - np.cos(half_left_rad) * left
    planes[2, :3] = n_left
    planes[2, 3] = -np.dot(n_left, T.ravel())

    # 右平面: 法向量 ⊥ 面, 指向内部（向左）, 过相机原点
    # 平面包含 right 边界射线 (cos(half_right), -sin(half_right), 0)_local 与 up 轴
    n_right = np.sin(half_right_rad
                    ) * forward + np.cos(half_right_rad) * left
    planes[3, :3] = n_right
    planes[3, 3] = -np.dot(n_right, T.ravel())

    # 上平面: 法向量 ⊥ 面, 指向内部（向下）, 过相机原点
    # 平面包含 top 边界射线 (cos(half_up), 0, sin(half_up))_local 与 left 轴
    n_top = np.sin(half_up_rad) * forward - np.cos(half_up_rad) * up
    planes[4, :3] = n_top
    planes[4, 3] = -np.dot(n_top, T.ravel())

    # 下平面: 法向量 ⊥ 面, 指向内部（向上）, 过相机原点
    # 平面包含 bottom 边界射线 (cos(half_down), 0, -sin(half_down))_local 与 left 轴
    n_bottom = np.sin(half_down_rad) * forward + np.cos(half_down_rad) * up
    planes[5, :3] = n_bottom
    planes[5, 3] = -np.dot(n_bottom, T.ravel())

    return planes


def are_points_in_frustum(
        points_xyz: np.ndarray,
        camera_pose: Pose,
        fov_h=None,
        fov_v=None,
        near: float = 0.1,
        far: float = 100.0
    ) -> np.ndarray:
    """
    判断点是否在相机视锥体内。

    Parameters
    ----------
    points_xyz : np.ndarray
        shape (N, 3) 或 (N, 2) float32, 世界坐标。
        如果为 (N, 2)，则 z=0 用于 2D 平面判断。
    camera_pose : Pose
        相机位姿
    fov_h, fov_v : float
        水平/垂直视场角
    near, far : float
        近/远裁剪面距离

    Returns
    -------
    np.ndarray
        shape (N,), dtype=bool, True 表示可见
    """
    if points_xyz.shape[1] == 2:
        pts = np.column_stack([points_xyz, np.zeros(len(points_xyz))])
    else:
        pts = points_xyz

    planes = compute_frustum_planes(camera_pose, fov_h, fov_v, near, far)

    # 每个点到各平面的有符号距离
    # plane: [a,b,c,d], 距离 = a*x + b*y + c*z + d
    # 法向量指向内部，所以距离 >= 0 表示在内部
    N = len(pts)
    # 扩展为齐次坐标
    pts_h = np.column_stack([pts, np.ones(N, dtype=np.float32)])  # (N, 4)
    distances = pts_h @ planes.T  # (N, 6)

    # 所有距离 >= 0 => 在视锥体内
    visible = np.all(distances >= 0, axis=1)
    return visible


def frustum_xy_bounds(
        camera_pose: Pose,
        fov_h=None,
        fov_v=None,
        near: float = 0.1,
        far: float = 100.0,
        margin: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回视锥顶点投影到 XOY 平面后的 AABB。

    用于在做完整视锥 / 深度投影前，先按全局 x/y 快速筛出候选点。
    """
    half_left, half_right = _to_fov_half_pair(fov_h, (45.0, 45.0))
    half_up, half_down = _to_fov_half_pair(fov_v, (30.0, 30.0))

    left_tan = np.tan(np.deg2rad(half_left))
    right_tan = np.tan(np.deg2rad(half_right))
    up_tan = np.tan(np.deg2rad(half_up))
    down_tan = np.tan(np.deg2rad(half_down))

    depths = np.array([max(float(near), 0.0), float(far)],
                        dtype=np.float32)
    corners = []
    for x in depths:
        for y_ratio in (-right_tan, left_tan):
            for z_ratio in (-down_tan, up_tan):
                corners.append([x, x * y_ratio, x * z_ratio])
    corners = np.asarray(corners, dtype=np.float32)
    corners_world = camera_pose.ego_to_wrd(corners)

    xy_min = np.min(corners_world[:, :2], axis=0) - margin
    xy_max = np.max(corners_world[:, :2], axis=0) + margin
    return xy_min.astype(np.float32), xy_max.astype(np.float32)


def frustum_culling(
    points_xyz: np.ndarray,
    camera_pose: Pose,
    fov_h=None,
    fov_v=None,
    near: float = 0.1,
    far: float = 100.0,
    min_distance: float = 0.0,
    max_distance: float = float("inf")
    ) -> np.ndarray:
    """
    综合可见性过滤（视锥体 + 距离范围）。

    Parameters
    ----------
    points_xyz : np.ndarray
        shape (N, 3) float32
    camera_pose : Pose
    fov_h : float or (float, float), optional
        水平视场角。标量时对称分配；tuple 时 (left_deg, right_deg)。
        默认 (45.0, 45.0) 即总 HFOV=90°。
    fov_v : float or (float, float), optional
        垂直视场角。标量时对称分配；tuple 时 (up_deg, down_deg)。
        默认 (30.0, 30.0) 即总 VFOV=60°。
    near, far : float
        裁剪面
    min_distance, max_distance : float
        距离范围过滤

    Returns
    -------
    np.ndarray
        shape (N,), dtype=bool
    """
    if points_xyz.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    visible = np.zeros(points_xyz.shape[0], dtype=bool)

    xy_min, xy_max = frustum_xy_bounds(
        camera_pose,
        fov_h=fov_h,
        fov_v=fov_v,
        near=near,
        far=far,
        )
    xy = points_xyz[:, :2]
    candidate_mask = ((xy[:, 0] >= xy_min[0])
                        & (xy[:, 0] <= xy_max[0])
                        & (xy[:, 1] >= xy_min[1])
                        & (xy[:, 1] <= xy_max[1]))
    candidate_idx = np.flatnonzero(candidate_mask)
    if candidate_idx.size == 0:
        return visible

    candidate_points = points_xyz[candidate_idx]

    in_frustum = are_points_in_frustum(
        candidate_points, camera_pose, fov_h, fov_v, near, far
        )

    # 距离过滤
    cam_pos = camera_pose.position.reshape(1, 3)
    if candidate_points.shape[1] == 2:
        dists = np.linalg.norm(candidate_points - cam_pos[:, :2], axis=1)
    else:
        dists = np.linalg.norm(candidate_points - cam_pos, axis=1)
    in_range = (dists >= min_distance) & (dists <= max_distance)

    visible[candidate_idx] = in_frustum & in_range
    return visible


def _pair2(v):
    """支持 (left, right) / (up, down)，也支持单个 float 表示左右/上下对称。"""
    if isinstance(v, (tuple, list, np.ndarray)):
        return float(v[0]), float(v[1])
    return float(v), float(v)


def _project_lidar_points(
        pts_lidar: np.ndarray, vertical_mode: str = "spherical"
    ):
    """
    雷达系: X=fwd, Y=left, Z=up.

    vertical_mode:
        "projective": pitch = atan2(z, x)，与 frustum_xy_bounds 里的 z/x 模型一致。
        "spherical":  pitch = atan2(z, sqrt(x^2 + y^2))，更接近部分机械/固态雷达定义。
    """
    pts = np.asarray(pts_lidar, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    r = np.linalg.norm(pts, axis=1)
    r_safe = np.maximum(r, 1e-6)

    yaw = np.rad2deg(np.arctan2(y, x))

    if vertical_mode == "spherical":
        horiz = np.sqrt(x*x + y*y)
        pitch = np.rad2deg(np.arctan2(z, np.maximum(horiz, 1e-6)))
    else:
        pitch = np.rad2deg(np.arctan2(z, np.maximum(x, 1e-6)))

    unit = pts / r_safe[:, None]
    return x, r, yaw, pitch, unit


def _frustum_mask_lidar(
        pts_lidar: np.ndarray,
        hfov=None,
        vfov=None,
        min_range: float = None,
        max_range: float = None,
        vertical_mode: str = "spherical"
    ):
    x, r, yaw, pitch, unit = _project_lidar_points(
        pts_lidar, vertical_mode=vertical_mode
    )
    selections = []
    if hfov is not None:
        h_left, h_right = _to_fov_half_pair(hfov)
        selections.append((yaw >= -h_right) & (yaw <= h_left))
    if vfov is not None:
        v_up, v_down = _to_fov_half_pair(vfov)
        selections.append((pitch >= -v_down) & (pitch <= v_up))
    if min_range is not None:
        selections.append(r >= float(min_range))
    if max_range is not None:
        selections.append(r <= float(max_range))
    if selections:
        mask = np.logical_and.reduce(selections)
    else:
        mask = np.ones_like(x, dtype=bool)
    return mask, r, yaw, pitch, unit


def _ratio(name, mask, total=None):
    if total is None:
        total = mask.size
    n = int(np.count_nonzero(mask))
    logger.debug(f"{name}: {n}/{total} = {n / max(total, 1):.3f}")


def remove_by_ray_carving(
        old_pts_lidar,
        old_r,
        old_unit,
        new_r,
        new_unit,
        *,
        voxel_size,
        hres,
        vres,
        k=64,
        beam_radius_factor=0.75,
        max_beam_radius=None,
        free_margin=None,
        workers=-1
    ):
    from scipy.spatial import cKDTree
    import numpy as np

    if old_pts_lidar.shape[0] == 0 or new_unit.shape[0] == 0:
        return np.zeros(old_pts_lidar.shape[0], dtype=bool)

    res_rad = np.deg2rad(max(float(hres), float(vres)))

    if max_beam_radius is None:
        max_beam_radius = 1.5 * voxel_size

    if free_margin is None:
        free_margin = max(0.5 * voxel_size, 0.03)

    # 米制射线管半径
    beam_radius = beam_radius_factor*voxel_size + old_r * np.tan(res_rad)
    beam_radius = np.clip(
        beam_radius,
        0.45 * voxel_size,
        max_beam_radius,
        )

    # 只用这个角度做候选召回，不直接做删除判定
    theta = np.arcsin(
        np.clip(beam_radius / np.maximum(old_r, 1e-4), 0.0, 0.95)
        )
    max_chord = 2.0 * np.sin(0.5 * float(np.max(theta)))

    tree = cKDTree(new_unit)
    dist, nn = tree.query(
        old_unit,
        k=int(k),
        # distance_upper_bound=max_chord,
        workers=workers,
    )

    if k == 1:
        dist = dist[:, None]
        nn = nn[:, None]

    n_new = new_unit.shape[0]
    valid = nn < n_new

    local_chord = 2.0 * np.sin(0.5 * theta)
    valid &= dist <= local_chord[:, None]

    nn_safe = np.where(valid, nn, 0)

    U = new_unit[nn_safe].astype(np.float32)
    R = new_r[nn_safe].astype(np.float32)
    P = old_pts_lidar.astype(np.float32)

    # 旧点在当前射线方向上的投影深度
    t = np.einsum("qkd,qd->qk", U, P)

    # 旧点到当前射线的横向距离
    d2 = old_r[:, None]**2 - t**2
    d_perp = np.sqrt(np.maximum(d2, 0.0))

    ray_crosses_old_voxel = (
        valid
        & (t > 0.0)
        & (d_perp <= beam_radius[:, None])
        )

    ray_says_free = ray_crosses_old_voxel & (t < R - free_margin)

    return np.any(ray_says_free, axis=1)


def compute_visible_empty_removal_mask(
        pts_world: np.ndarray,
        pts_new: np.ndarray,
        pose_lidar,
        hfov_update,
        vfov_update,
        hres: float,
        vres: float,
        min_range: float,
        max_range: float,
        *,
        hfov_filter=None,
        vfov_filter=None,
        candidate_idx: np.ndarray | None = None,
        voxel_size: float = 0.10,
        vertical_mode: str = "spherical",
        k: int = 9,
        min_support: int = 4,
        angle_gate_res_factor: float = 2.0,
        angle_gate_voxel_factor: float = 1.5,
        max_angle_gate_deg: float = 60.0,
        free_base_margin: float = 0.02,
        free_voxel_factor: float = 0.3,
        plane_rmse_abs: float = 0.8,
        plane_rmse_rel: float = 0.02,
        min_ray_plane_cos: float = 0.05,
        require_angular_bracket: bool = False,
        workers: int = -1,
        world_to_lidar=None,
        debug=False
    ) -> np.ndarray:
    """
    返回 bool mask，shape = (len(pts_world),)
    True 表示该全局旧点在当前视角下被判定为空，应剔除。

    pts_world: 全局历史点，世界系，(N, 3)
    pts_new: 当前帧观测点，世界系，(M, 3)
    pose_lidar: 推荐传入 world->lidar 的 Pose
    candidate_idx: 可选，AABB 粗筛后的全局点索引
    """

    pts_world = np.asarray(pts_world, dtype=np.float32)
    pts_new = np.asarray(pts_new, dtype=np.float32)

    n_world = pts_world.shape[0]
    remove_mask = np.zeros(n_world, dtype=bool)

    if pts_new.shape[0] < min_support or n_world == 0:
        return remove_mask

    if candidate_idx is None:
        candidate_idx = np.arange(n_world, dtype=np.int64)
    else:
        candidate_idx = np.asarray(candidate_idx, dtype=np.int64)

    if candidate_idx.size == 0:
        return remove_mask

    if world_to_lidar is None:
        world_to_lidar = ~pose_lidar

    # 1. 世界系 -> 雷达系
    pts_new_lidar = world_to_lidar(pts_new).astype(np.float32, copy=False)
    pts_old_lidar = world_to_lidar(pts_world[candidate_idx]).astype(
        np.float32, copy=False
        )

    # 2. 当前帧点完整视锥裁剪
    new_valid, r_new_all, yaw_new_all, pitch_new_all, unit_new_all = _frustum_mask_lidar(
        pts_new_lidar,
        hfov=hfov_filter,
        vfov=vfov_filter,
        min_range=min_range,
        max_range=max_range,
        vertical_mode=vertical_mode,
    )

    if np.count_nonzero(new_valid) < min_support:
        return remove_mask

    new_pts = pts_new_lidar[new_valid]
    new_r = r_new_all[new_valid]
    new_yaw = yaw_new_all[new_valid]
    new_pitch = pitch_new_all[new_valid]
    new_unit = unit_new_all[new_valid]

    # 3. 历史候选点完整视锥裁剪
    old_valid, r_old_all, yaw_old_all, pitch_old_all, unit_old_all = _frustum_mask_lidar(
        pts_old_lidar,
        hfov=hfov_update,
        vfov=vfov_update,
        min_range=min_range,
        max_range=max_range,
        vertical_mode=vertical_mode,
    )

    if np.count_nonzero(old_valid) == 0:
        return remove_mask

    cand_valid_idx = candidate_idx[old_valid]
    old_unit = unit_old_all[old_valid]
    old_r = r_old_all[old_valid]
    old_yaw = yaw_old_all[old_valid]
    old_pitch = pitch_old_all[old_valid]

    # 4. 当前帧方向 KDTree
    tree = cKDTree(new_unit)

    # chord distance = ||u1 - u2|| = 2 * sin(theta / 2)
    res_rad = np.deg2rad(max(float(hres), float(vres)))
    theta_base = angle_gate_res_factor * res_rad

    # 近距离处，一个 0.1m 体素对应更大的角度；远距离主要由雷达角分辨率决定
    theta_local = theta_base + np.arctan2(
        angle_gate_voxel_factor * voxel_size,
        np.maximum(old_r, 1e-3),
        )
    theta_local = np.minimum(theta_local, np.deg2rad(max_angle_gate_deg))

    theta_global = float(np.max(theta_local))
    chord_global = 2.0 * np.sin(0.5 * theta_global)

    dist, nn = tree.query(
        old_unit,
        k=int(k),
        distance_upper_bound=chord_global,
        workers=workers,
    )

    if k == 1:
        dist = dist[:, None]
        nn = nn[:, None]

    n_new = new_pts.shape[0]

    # SciPy 缺失邻居用 idx == n_new 和 dist == inf 表示
    nn_valid = nn < n_new

    # 再做 per-query 的局部角阈值过滤
    chord_local = 2.0 * np.sin(0.5 * theta_local)
    nn_valid &= dist <= chord_local[:, None]

    support_count = np.sum(nn_valid, axis=1)
    enough_support = support_count >= int(min_support)

    if not np.any(enough_support):
        return remove_mask

    # 5. gather 支持点
    nn_safe = np.where(nn_valid, nn, 0)

    P = new_pts[nn_safe].astype(np.float64)  # (Q, k, 3)
    R = new_r[nn_safe].astype(np.float64)  # (Q, k)
    Yaw = new_yaw[nn_safe].astype(np.float64)
    Pitch = new_pitch[nn_safe].astype(np.float64)

    W = nn_valid.astype(np.float64)
    cnt = np.maximum(support_count.astype(np.float64), 1.0)

    # 6. 可选：要求支持点在角域上不要全在查询方向一侧，避免边缘外推误删
    if require_angular_bracket:
        dyaw = ((Yaw - old_yaw[:, None] + 180.0) % 360.0) - 180.0
        dpitch = Pitch - old_pitch[:, None]

        has_yaw_neg = np.any((dyaw <= 0.0) & nn_valid, axis=1)
        has_yaw_pos = np.any((dyaw >= 0.0) & nn_valid, axis=1)
        has_pitch_neg = np.any((dpitch <= 0.0) & nn_valid, axis=1)
        has_pitch_pos = np.any((dpitch >= 0.0) & nn_valid, axis=1)

        nearest_chord = np.min(np.where(nn_valid, dist, np.inf), axis=1)
        near_center = nearest_chord <= 2.0 * np.sin(0.5 * res_rad)

        angular_ok = (((has_yaw_neg & has_yaw_pos) | near_center)
                        & ((has_pitch_neg & has_pitch_pos) | near_center))
    else:
        angular_ok = np.ones_like(enough_support, dtype=bool)

    # 7. 批量局部平面拟合
    mu = np.sum(P * W[:, :, None], axis=1) / cnt[:, None]
    Q = (P - mu[:, None, :]) * W[:, :, None]

    C = np.einsum("qki,qkj->qij", Q, Q) / cnt[:, None, None]

    eigvals, eigvecs = np.linalg.eigh(C)
    normal = eigvecs[:, :, 0]  # 最小特征值方向

    plane_rmse = np.sqrt(np.maximum(eigvals[:, 0], 0.0))

    old_unit64 = old_unit.astype(np.float64)
    den = np.sum(normal * old_unit64, axis=1)
    num = np.sum(normal * mu, axis=1)

    t_surface = np.full_like(old_r, np.nan, dtype=np.float64)
    stable_intersection = np.abs(den) >= float(min_ray_plane_cos)
    t_surface[stable_intersection
             ] = num[stable_intersection] / den[stable_intersection]

    # 8. 自适应自由空间 margin
    # 15m, 0.36deg 时，角分辨率带来的横向不确定性约为 15*tan(0.36deg) ~= 0.094m
    tau_free = (
        float(free_base_margin)
        + float(free_voxel_factor) * float(voxel_size)
        + old_r.astype(np.float64) * np.tan(res_rad)
        )

    rmse_gate = np.maximum(
        float(plane_rmse_abs),
        float(plane_rmse_rel) * np.maximum(t_surface, 1.0),
        )

    plane_ok = (
        enough_support
        & angular_ok
        & stable_intersection
        & np.isfinite(t_surface)
        & (t_surface >= float(min_range))
        & (t_surface <= float(max_range) * 1.05)
        # & (plane_rmse <= rmse_gate)
        )

    # 9. 主判定：历史点位于当前观测面前方的自由空间
    remove_by_plane = plane_ok & (
        old_r.astype(np.float64) < t_surface - tau_free
        )

    # 10. 保守 fallback：
    # 如果平面不可靠，但所有近邻当前点都明显比旧点更远，也可认为旧点前方为空。
    # 这个分支比平面更保守，要求支持点更多。
    min_support_depth = np.min(np.where(nn_valid, R, np.inf), axis=1)

    old_pts_valid_lidar = pts_old_lidar[old_valid]
    # remove_ray = remove_by_ray_carving(
    #     old_pts_lidar=old_pts_valid_lidar,
    #     old_r=old_r,
    #     old_unit=old_unit,
    #     new_r=new_r,
    #     new_unit=new_unit,
    #     voxel_size=voxel_size,
    #     hres=hres,
    #     vres=vres,
    #     k=64,
    #     beam_radius_factor=0.75,
    #     max_beam_radius=1.5 * voxel_size,
    #     free_margin=max(0.5 * voxel_size, 0.03),
    #     workers=workers,
    #     )

    remove_by_depth_envelope = (
        (~plane_ok)
        & enough_support
        & angular_ok
        & (support_count >= max(min_support, k // 2))
        & np.isfinite(min_support_depth)
        & (old_r.astype(np.float64) < min_support_depth - tau_free)
        )

    remove_local = remove_by_plane | remove_by_depth_envelope

    # remove_local = remove_ray | ((old_r > 1.5) & remove_by_plane
    #                             )  # | remove_by_depth_envelope

    remove_mask[cand_valid_idx] = remove_local

    if debug:
        logger.debug("---- visible empty debug ----")
        logger.debug(f"pts_world: {pts_world.shape}")
        logger.debug(f"pts_new: {pts_new.shape}")
        logger.debug(f"candidate_idx: {candidate_idx.shape}")

        logger.debug(
            f"new lidar x percentiles: {np.percentile(pts_new_lidar[:, 0], [0, 1, 5, 50, 95, 99, 100])}",
            )
        logger.debug(
            f"old lidar x percentiles: {np.percentile(pts_old_lidar[:, 0], [0, 1, 5, 50, 95, 99, 100])}",
            )

        _ratio("new_valid", new_valid)
        _ratio("old_valid", old_valid)

        tmp = np.percentile(
            r_new_all[new_valid], [0, 1, 5, 50, 95, 99, 100]
            ) if np.any(new_valid) else None
        logger.debug(f"new range percentiles: {tmp}")
        tmp = np.percentile(
            r_old_all[old_valid], [0, 1, 5, 50, 95, 99, 100]
            ) if np.any(old_valid) else None
        logger.debug(f"old range percentiles: {tmp}")
        tmp = np.percentile(
            support_count, [0, 1, 5, 25, 50, 75, 95, 99, 100]
            ) if np.any(support_count) else None
        logger.debug(f"support_count percentiles: {tmp}")
        _ratio("enough_support", enough_support)
        _ratio("angular_ok", angular_ok)
        _ratio("stable_intersection", stable_intersection)
        _ratio("plane_ok", plane_ok)
        _ratio("remove_by_plane", remove_by_plane)
        _ratio("remove_by_depth_envelope", remove_by_depth_envelope)
        _ratio("remove_local", remove_local)
        valid_depth_cmp = np.isfinite(min_support_depth)
        delta = min_support_depth[valid_depth_cmp] - old_r[valid_depth_cmp]
        tmp = np.percentile(delta, [0, 1, 5, 25, 50, 75, 95, 99, 100
                                   ]) if np.any(delta) else None
        logger.debug(
            f"nearest/support current depth - old depth percentiles: {tmp}"
            )
        logger.debug(f"positive depth gap ratio: {np.mean(delta > 0.05)}")
    return remove_mask


def get_surround_points_2d(
        points: np.ndarray,
        pos_ego: np.ndarray,
        res_voxel: float = None,
        safe_range: float = None,
        res_deg: float = None
    ):
    """
    Returns
    -------
        filtered_points : (M, 2)
            降采样后的2D点云
        mask : (N,)
            bool类型掩码，表示哪些原始点被保留
    """
    # 0. 判空保护
    if points is None or len(points) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=bool)

    # 1. 确定实际的角分辨率 (Degree)
    if res_deg is None:
        if res_voxel is not None and safe_range is not None:
            # 动态计算角分辨率：确保在 safe_range 处的弧长 <= res_voxel
            res_rad = res_voxel / safe_range
            res_deg = np.degrees(res_rad)
        else:
            raise ValueError(
                "必须明确指定 res_deg，或者同时指定 res_voxel 和 safe_range"
                )

    # 为了让圆周完美闭合，计算实际需要划分的桶(bins)的总数，并倒推修正后的角分辨率
    num_bins = max(1, int(np.round(360.0 / res_deg)))
    actual_res_deg = 360.0 / num_bins

    # 2. 计算相对坐标、距离(极径)和方位角(极角)
    diff = points - pos_ego
    # 使用 np.hypot 比 np.linalg.norm 更快
    dists = np.hypot(diff[:, 0], diff[:, 1])
    angles = np.arctan2(diff[:, 1], diff[:, 0])  # 范围: [-pi, pi]

    # 3. 计算每个点所属的桶索引 (Bin Index)
    # 将角度转为度数并除以实际角分辨率，四舍五入后对桶数取模保证在 [0, num_bins-1]
    bin_indices = np.round(np.degrees(angles) / actual_res_deg) % num_bins
    bin_indices = bin_indices.astype(np.int32)

    # 4. 高效选取每个桶中距离最近的点 (核心提速区)
    # 方案: 对距离进行排序，然后通过 np.unique 获取每个桶首次出现的索引(即最近点的索引)
    sort_idx = np.argsort(dists)
    sorted_bins = bin_indices[sort_idx]

    # np.unique 的 return_index=True 会返回每个唯一元素【第一次】出现的索引
    # 因为我们已经按距离升序排列了，所以拿到的必然是该方位角桶里距离 ego 最近的那个点！
    _, unique_idx = np.unique(sorted_bins, return_index=True)

    # 获取需要保留的原始点云索引
    keep_indices = sort_idx[unique_idx]

    # 5. 构建输出
    # 生成 Boolean 掩码 (或者您也可以直接返回 keep_indices)
    mask = np.zeros(len(points), dtype=bool)
    mask[keep_indices] = True

    filtered_points = points[keep_indices]

    return filtered_points, mask
