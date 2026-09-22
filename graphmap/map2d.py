"""
Map2D — 2D 可行地图（遍历性栅格）

基于固定分辨率的二维栅格，每个格存储 float32 值（-1 = 未知, 0 = 障碇, 1 = 可行）。
支持世界坐标 ↔ 栅格坐标转换，以及基于切片/数组的索引访问。

可行区域判定:
    - update_from_occupancy:  从 3D 点云标记障碇物 (=0)
    - update_free_by_raycast: 射线投射标记可见可行区域 (=1)
    - update_traversable_from_ground: 地面点云验证可行区域 (=1)
    - update_traversable:     组合方法，含地面空洞防护
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, List
import io
import cv2
from .pose import Pose
from .misc.config import Munch

from scipy.ndimage import binary_dilation


class Map2D:

    def __init__(
            self,
            config: Optional[Munch] = None,
            resolution: float = 0.1,
            side_length: float = 100.0,
            num_layers: int = 1,
            shape: Optional[Tuple[int, int]] = None,
            origin: Optional[Tuple[float, float]] = None,
            **kwds
        ):
        """
        Parameters
        ----------
        resolution : float
            每个栅格的边长，单位：米
        side_length : float
            地图总边长，单位：米（正方形地图）
        shape : tuple, optional
            显式栅格尺寸 (H, W)。提供后允许矩形地图。
        num_layers : int
            地图层数（默认 1）
        origin : tuple, optional
            世界原点对应的栅格坐标 (h, w) / (x_index, y_index)，默认为地图中心
        """
        self.config = config
        map_config = getattr(config, "map", None)
        configured_resolution = (
            getattr(map_config, "resolution", None)
            if map_config is not None else None
            )
        configured_side_length = (
            getattr(map_config, "side_length", None)
            if map_config is not None else None
            )
        self.resolution = float(configured_resolution or resolution)
        self.side_length = float(configured_side_length or side_length)
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.side_length <= 0:
            raise ValueError("side_length must be positive")
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")

        if shape is None:
            w = int(np.ceil(self.side_length / self.resolution))
            h = w
        else:
            if len(shape) != 2:
                raise ValueError("shape must contain (height, width)")
            w, h = (int(shape[0]), int(shape[1]))
            if w <= 0 or h <= 0:
                raise ValueError("shape dimensions must be positive")
        self._grid = np.full((w, h, num_layers), -1.0, dtype=np.float32)

        if origin is not None:
            self._origin = np.array(origin, dtype=np.float64)
        else:
            self._origin = np.array([w / 2.0, h / 2.0], dtype=np.float64)

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def grid(self) -> np.ndarray:
        """shape (H, W, L) 栅格数组。"""
        return self._grid

    @property
    def shape(self) -> Tuple[int, int, int]:
        """(H, W, L)。"""
        return self._grid.shape

    @property
    def origin(self) -> np.ndarray:
        """世界原点在栅格中的坐标 (h, w) / (x_index, y_index)。"""
        return self._origin

    # ── 坐标转换 ──────────────────────────────────────────────────────

    def world_to_grid(self, coords: np.ndarray) -> np.ndarray:
        """
        世界坐标 → 栅格索引。

        Parameters
        ----------
        coords : np.ndarray
            shape (N, 2) [x, y] 或 shape (2,) [x, y]

        Returns
        -------
        np.ndarray
            shape (N, 2) 或 (2,) int32 栅格索引 [h, w] = [x_index, y_index]
        """
        coords = np.asarray(coords, dtype=np.float64)
        is_1d = coords.ndim == 1
        if is_1d:
            coords = coords.reshape(1, 2)
        grid_idx = np.round(
            coords[:, :2] / self.resolution + self._origin[None, :2]
            ).astype(np.int32)
        if is_1d:
            return grid_idx[0]
        return grid_idx

    def grid_to_world(self, indices: np.ndarray) -> np.ndarray:
        """
        栅格索引 → 世界坐标。

        Parameters
        ----------
        indices : np.ndarray
            shape (N, 2) 或 (2,) int32 [h, w] = [x_index, y_index]

        Returns
        -------
        np.ndarray
            shape (N, 2) 或 (2,) float64 [x, y]
        """
        indices = np.asarray(indices, dtype=np.int32)
        is_1d = indices.ndim == 1
        if is_1d:
            indices = indices.reshape(1, 2)
        world = (indices - self._origin[None, :2]) * self.resolution
        if is_1d:
            return world[0]
        return world

    def is_in_bounds(self, indices: np.ndarray) -> np.ndarray:
        """检查栅格索引是否在地图范围内。"""
        indices = np.asarray(indices, dtype=np.int32)
        w, h = self._grid.shape[0], self._grid.shape[1]
        return ((indices[..., 0] >= 0) & (indices[..., 0] < w) &
                (indices[..., 1] >= 0) & (indices[..., 1] < h))

    # ── 索引访问（世界坐标） ───────────────────────────────────────────

    def __getitem__(self, key):
        """
        支持世界坐标下的 NumPy 风格索引。

        示例:
            map[1.0:3.0, 0.5:2.0]     # 切片
            map[0.3, 0.4]             # 单点
            map[[0.1, 0.2], [0.3, 0.4]] # 批量
        """
        return self._grid[self._process_index(key)]

    def __setitem__(self, key, value):
        self._grid[self._process_index(key)] = value

    def _process_index(self, key):
        """将世界坐标索引转换为栅格索引元组。"""
        if isinstance(key, tuple):
            # 提取 layer
            main_key = key[:2]
            layer = key[2:] if len(key) > 2 else (slice(None), )
            xy_idx = self.world_to_grid_sequence(main_key)
            return xy_idx + layer
        else:
            return self.world_to_grid_sequence((key, ))

    def world_to_grid_sequence(self, seq):
        """转换 (arr_x, arr_y) 序列为栅格索引序列。"""
        tmp = []
        for i, item in enumerate(seq):
            if isinstance(item, slice):
                g_start = None if item.start is None else int(
                    np.round(item.start / self.resolution + self._origin[i])
                    )
                g_stop = None if item.stop is None else int(
                    np.round(item.stop / self.resolution + self._origin[i])
                    )
                g_step = None if item.step is None else int(
                    np.round(item.step / self.resolution)
                    )
                if g_step == 0:
                    raise ValueError("slice step must span at least one grid cell")
                tmp.append(slice(g_start, g_stop, g_step))
            else:
                idx = np.asarray(item)
                grid_idx = np.round(
                    (idx / self.resolution + self._origin[i])
                    ).astype(np.int32)
                tmp.append(grid_idx)
        return tuple(tmp)

    # ── 批量更新 ──────────────────────────────────────────────────────

    def update(
            self,
            pose_robot: Pose,
            points_obstacle: np.ndarray,
            points_floor: np.ndarray,
            **kwds
        ):
        pos = pose_robot.position[:2]
        yaw = pose_robot.euler_rad[2]

    def update_points(
            self,
            points: np.ndarray,
            layer: int = 0,
            value: Optional[Union[float, np.ndarray]] = None
        ):
        """
        用点批量更新地图。

        Parameters
        ----------
        points : np.ndarray
            shape (N, 2) 世界坐标 [x, y] 或 shape (N, 3) [x, y, value]
        layer : int
            更新的图层索引
        value : float or np.ndarray, optional
            统一值或逐点值。不提供则使用 points[:, 2]
        """
        loc = points[:, :2]
        loc_grid = self.world_to_grid(loc)
        # 过滤越界
        valid = self.is_in_bounds(loc_grid)
        loc_grid = loc_grid[valid]

        if value is not None:
            val = np.broadcast_to(value, len(points))[valid]
        elif points.shape[1] >= 3:
            val = points[valid, 2]
        else:
            raise ValueError(
                "No value provided and points has no 3rd column"
                )

        self._grid[loc_grid[:, 0], loc_grid[:, 1], layer] = val

    def update_from_occupancy(self, points: np.ndarray, layer: int = 0):
        """
        从 3D 点云 (x,y,z) 更新占用地图 (=0)。默认投影所有点；
        若 config.agent 提供 collision_height_min/max，则按其高度区间筛选。

        Parameters
        ----------
        xyz : np.ndarray
            shape (N, 3) [x, y, z]
        layer : int
            图层索引
        """
        agent_config = getattr(self.config, "agent", None)
        collision_height_min = getattr(agent_config, "collision_height_min", -np.inf)
        collision_height_max = getattr(agent_config, "collision_height_max", np.inf)
        obstacle = ((points[:, 2] > collision_height_min) &
                    (points[:, 2] < collision_height_max))
        obstacle_pts = points[obstacle]
        if len(obstacle_pts) > 0:
            self.update_points(obstacle_pts, layer=layer, value=0.0)

    def update_free_by_polygon(
            self,
            robot_pos: np.ndarray,
            robot_yaw: float,
            points_obstacle: np.ndarray,
            robot_radius: float = 0.3,
            hfov_deg: tuple = (45.0, 45.0),
            max_range: float = 10.0,
            layer: int = 0,
            angle_res_deg: float = 0.5
        ):
        """
        包含 HFOV 视场角填充 与 自身半径强制清空的 3D转2D 栅格更新算法。
        """
        robot_radius = robot_radius or self.config.agent.radius

        robot_pos = np.asarray(robot_pos, dtype=np.float64).ravel()[:2]
        robot_grid = self.world_to_grid(robot_pos)

        if not self.is_in_bounds(robot_grid):
            return

        w, h = self._grid.shape[0], self._grid.shape[1]

        # 预先在内存分配整张地图大小的掩膜
        mask = np.zeros((w, h), dtype=np.uint8)

        # ==========================================
        # 1. 3D 高度滤波 (Z-axis Pass-through)
        # ==========================================
        pts = np.asarray(points_obstacle, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        else:
            pts = pts[:, :2]

        # ==========================================
        # 2. 距离计算与过滤
        # ==========================================
        dists = np.array([])
        angles = np.array([])
        if len(pts) > 0:
            delta = pts - robot_pos
            dists = np.linalg.norm(delta, axis=1)
            # 滤除超视距的点
            within_range = dists <= max_range
            delta = delta[within_range]
            dists = dists[within_range]
            pts = pts[within_range]

            if len(dists) > 0:
                angles = np.arctan2(
                    delta[:, 1], delta[:, 0]
                    )  # 世界坐标系下的极角 [-pi, pi]

        # ==========================================
        # 3. 极坐标分桶与 HFOV 视场角逻辑注入 (核心新增)
        # ==========================================
        num_bins = int(np.ceil(360.0 / angle_res_deg))

        # 初始化所有桶的距离为无穷大
        min_dists = np.full(num_bins, np.inf, dtype=np.float64)

        # 如果有真实的雷达点，把它们打入对应的桶中找最小值
        if len(dists) > 0:
            angles_norm = angles + np.pi
            bin_idx = np.floor((angles_norm /
                                (2 * np.pi)) * num_bins).astype(np.int32)
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)
            np.minimum.at(min_dists, bin_idx, dists)

        # 计算每个桶在世界坐标系下的中心角度
        bin_angles_world = (np.arange(num_bins)
                            + 0.5) * (2 * np.pi / num_bins) - np.pi

        # 【关键】将世界角度转换为相对于机器人车头 (yaw) 的相对角度 [-pi, pi]
        bin_rel_angles = (bin_angles_world - robot_yaw
                            + np.pi) % (2 * np.pi) - np.pi

        # 解析 HFOV (左偏角, 右偏角) 并转换为弧度
        hfov_left_rad = np.deg2rad(hfov_deg[0])  # 正值表示向左
        hfov_right_rad = np.deg2rad(hfov_deg[1])  # 正值表示向右，在轴上属于负方向区间

        # 筛选出落在视场角内的桶掩码
        in_fov_mask = (bin_rel_angles >= -hfov_right_rad
                      ) & (bin_rel_angles <= hfov_left_rad)

        # 【新增功能2】：对于视场角内部、但是没有扫到任何点 (距离为inf) 的桶，将其设为最大可视距离 (max_range)
        empty_in_fov = in_fov_mask & np.isinf(min_dists)
        min_dists[empty_in_fov] = max_range

        # （可选）严格模式：将视场角以外的干扰噪点全部丢弃，不予绘制
        # min_dists[~in_fov_mask] = np.inf

        # ==========================================
        # 4. 构建扫描多边形
        # ==========================================
        valid_bins = np.where(~np.isinf(min_dists))[0]

        if len(valid_bins) > 0:
            valid_angles_world = bin_angles_world[valid_bins]
            valid_angles_rel = bin_rel_angles[valid_bins]
            valid_dists = min_dists[valid_bins]

            # 【细节魔法】：按相对角度重新排序，确保多边形的顶点是按顺序连贯扫描的。
            # 否则当视场角跨越世界坐标系的边界 (如从 -180度跳跃到 180度) 时，图形内部会出现交叉切割的错乱连线
            sort_idx = np.argsort(valid_angles_rel)
            valid_angles_world = valid_angles_world[sort_idx]
            valid_dists = valid_dists[sort_idx]

            # 转笛卡尔坐标系
            poly_x = robot_pos[0
                              ] + valid_dists * np.cos(valid_angles_world)
            poly_y = robot_pos[1
                              ] + valid_dists * np.sin(valid_angles_world)

            # 加入机器人中心作为多边形的原点起点闭环
            poly_pts_world = np.column_stack((poly_x, poly_y))
            poly_pts_world = np.vstack((robot_pos, poly_pts_world))

            # 坐标系转换与填充
            poly_pts_grid = self.world_to_grid(poly_pts_world)
            cv_pts = poly_pts_grid[:, [1, 0]].reshape((-1, 1, 2)
                                                     ).astype(np.int32)
            cv2.fillPoly(mask, [cv_pts], 1)

        # ==========================================
        # 5. 障碍物端点保护 (从mask中挖掉障碍物)
        # ==========================================
        if len(pts) > 0:
            obs_grid = self.world_to_grid(pts)
            valid_obs = self.is_in_bounds(obs_grid)
            obs_grid = obs_grid[valid_obs]
            if len(obs_grid) > 0:
                mask[obs_grid[:, 0], obs_grid[:, 1]] = 0

        # ==========================================
        # 6. 机器人自身半径清空 (强制覆写覆盖)
        # ==========================================
        cols_robot, rows_robot = int(robot_grid[0]), int(robot_grid[1])
        mask[cols_robot, rows_robot] = 1
        radius_px = int(np.ceil(robot_radius / self.resolution))
        if radius_px > 0:
            cv2.circle(
                mask, (rows_robot, cols_robot), radius_px, 1, thickness=-1
                )

        # ==========================================
        # 7. 将计算完毕的 Mask 批量刷入底层地图图层
        # ==========================================
        self._grid[mask == 1, layer] = 1.0

    def update_traversable_from_ground(
            self,
            ground_xyz: np.ndarray,
            z_range: "Optional[Tuple[float, float]]" = None,
            layer: int = 0
        ):
        """
        基于地面点云标记可行区域。

        仅在 ground_xyz 点所在的栅格处标记为可行 (=1)。
        不会标记没有地面观测的栅格，从而防止地面空洞被误纳入可行区域。
        不会覆盖已存在的障碍物标记 (=0)。

        Parameters
        ----------
        ground_xyz : np.ndarray
            shape (N, 3) [x, y, z] 地面点云世界坐标
        z_range : (float, float), optional
            Z 轴范围过滤，如 (-0.5, 0.5)。不提供则不过滤。
        layer : int
            图层索引
        """
        pts = np.asarray(ground_xyz, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[1] < 3:
            raise ValueError(
                "ground_xyz must have at least 3 columns [x, y, z]"
                )

        if z_range is not None:
            z_mask = (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])
            pts = pts[z_mask]

        if len(pts) == 0:
            return

        # 转栅格坐标，过滤越界 & 已标记障碍物的栅格
        grid_idx = self.world_to_grid(pts[:, :2])
        valid = self.is_in_bounds(grid_idx)
        grid_idx = grid_idx[valid]
        pts = pts[valid]
        if len(pts) == 0:
            return

        # 不覆盖已有的障碍物 (=0)
        not_obstacle = self._grid[grid_idx[:, 0], grid_idx[:, 1],
                                    layer] != 0.0
        pts = pts[not_obstacle]
        if len(pts) == 0:
            return

        self.update_points(pts, layer=layer, value=1.0)

    def update_traversable(
            self,
            robot_pos: np.ndarray,
            observed_xyz: np.ndarray,
            ground_xyz: "Optional[np.ndarray]" = None,
            *,
            collision_z_min: float = 0.2,
            collision_z_max: float = 2.0,
            max_range: float = 10.0,
            ground_search_radius: float = 0.5,
            ground_z_range: "Optional[Tuple[float, float]]" = None,
            layer: int = 0
        ):
        """
        组合式可行区域更新：障碍物标记 + 射线投射 + 地面验证。

        流程:
            1. 从 observed_xyz 筛选碰撞高度内的点为障碍物，标记为 (=0)
            2. 射线投射：机器人到每个障碍物连线 → 标记为可行 (=1)
            3. 若提供 ground_xyz，地面点处标记为可行 (=1)
            4. 若提供 ground_xyz，移除没有地面支撑的射线可行区域
               （防止地面空洞被误纳入可行区域）

        Parameters
        ----------
        robot_pos : np.ndarray
            shape (2,) [x, y] 机器人世界坐标
        observed_xyz : np.ndarray
            shape (N, 3) [x, y, z] 传感器观测到的 3D 点云
        ground_xyz : np.ndarray, optional
            shape (M, 3) [x, y, z] 地面点云。提供后用于验证可行走性。
        collision_z_min : float
            障碍物高度下界（米，相对地面）
        collision_z_max : float
            障碍物高度上界（米，相对地面）
        max_range : float
            射线最大距离（米）
        ground_search_radius : float
            地面验证搜索半径（米）。射线可行区域中，若某个栅格在此半径内
            没有地面观测点，则从可行区域中移除（防止空洞）。
        ground_z_range : (float, float), optional
            地面点 Z 轴范围过滤
        layer : int
            图层索引
        """
        robot_pos = np.asarray(robot_pos, dtype=np.float64).ravel()[:2]
        pts = np.asarray(observed_xyz, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        # --- 1. 筛选碰撞高度内的障碍点 ---
        z = pts[:, 2]
        collision_mask = (z > collision_z_min) & (z < collision_z_max)
        collision_xyz = pts[collision_mask]

        # --- 2. 标记障碍物 ---
        if len(collision_xyz) > 0:
            self.update_points(collision_xyz, layer=layer, value=0.0)

        # --- 3. 记录射线投射前的可行区域（用于后续区分新增区域） ---
        prev_free = (self._grid[:, :, layer] == 1.0).copy()

        # --- 4. 射线投射标记可行区域 ---
        self.update_free_by_raycast(
            robot_pos, collision_xyz[:, :2], max_range, layer
            )

        # --- 5. 地面点云标记可行区域 + 空洞防护 ---
        if ground_xyz is not None and len(ground_xyz) > 0:
            # 5a. 标记地面点为可行
            ground = np.asarray(ground_xyz, dtype=np.float64)
            if ground.ndim == 1:
                ground = ground.reshape(1, -1)
            if ground_z_range is not None:
                gz_mask = (ground[:, 2] >= ground_z_range[0]
                          ) & (ground[:, 2] <= ground_z_range[1])
                ground = ground[gz_mask]

            if len(ground) > 0:
                # 地面点直接标记为可行
                ground_grid = self.world_to_grid(ground[:, :2])
                valid = self.is_in_bounds(ground_grid)
                ground_grid = ground_grid[valid]
                if len(ground_grid) > 0:
                    # 不覆盖已标记的障碍物栅格
                    not_obstacle = self._grid[ground_grid[:, 0],
                                                ground_grid[:, 1],
                                                layer] != 0.0
                    safe_grid = ground_grid[not_obstacle]
                    if len(safe_grid) > 0:
                        self._grid[safe_grid[:, 0], safe_grid[:, 1],
                                    layer] = 1.0

                # 5b. 构建地面二值图（有地面观测 → True）
                w, h = self._grid.shape[0], self._grid.shape[1]
                ground_binary = np.zeros((w, h), dtype=bool)
                ground_binary[ground_grid[:, 0], ground_grid[:, 1]] = True

                # 5c. 膨胀 ground_search_radius 范围
                dilate_pixels = max(
                    1,
                    int(np.ceil(ground_search_radius / self.resolution))
                    )
                struct = np.ones(
                    (2*dilate_pixels + 1, 2*dilate_pixels + 1), dtype=bool
                    )
                ground_dilated = binary_dilation(
                    ground_binary, structure=struct
                    )

                # 5d. 找出射线投射新增的可行区域
                new_free = (self._grid[:, :, layer] == 1.0) & ~prev_free

                # 5e. 移除没有地面支撑的新可行区域（防止空洞）
                unsupported = new_free & ~ground_dilated
                self._grid[unsupported, layer] = -1.0

    # ── 序列化 ────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path], to_bytes: bool = False):
        """
        保存地图。

        Parameters
        ----------
        path : str or Path
            保存路径
        to_bytes : bool
            如果为 True，返回 bytes 而不写入文件
        """
        values = {
            "grid":
                self._grid,
            "origin":
                self._origin,
            "resolution":
                np.array(self.resolution, dtype=np.float64),
            "side_length":
                np.array(self.side_length, dtype=np.float64),
            "shape":
                np.asarray(self._grid.shape[:2], dtype=np.int64),
            "num_layers":
                np.array(self._grid.shape[2], dtype=np.int64),
            # Keep legacy readers working.
            "metadata":
                np.array(
                    [
                        self.resolution,
                        self.side_length,
                        self._grid.shape[2]
                        ],
                    dtype=np.float64,
                    ),
            }
        if to_bytes:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **values)
            return buffer.getvalue()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **values)

    @classmethod
    def load(
            cls,
            path: Union[str, Path] = None,
            bytes_data: bytes = None
        ) -> "Map2D":
        """
        加载地图。

        Parameters
        ----------
        path : str or Path, optional
            文件路径
        bytes_data : bytes, optional
            从内存加载
        """
        if bytes_data is not None:
            buffer = io.BytesIO(bytes_data)
            loader = np.load(buffer)
        else:
            loader = np.load(path)

        with loader as data:
            if "resolution" in data:
                resolution = float(data["resolution"])
                side_length = float(data["side_length"])
                shape = tuple(int(value) for value in data["shape"])
                num_layers = int(data["num_layers"])
            else:
                metadata = data["metadata"]
                resolution = float(metadata[0])
                side_length = float(metadata[1])
                num_layers = int(metadata[2])
                shape = tuple(
                    int(value) for value in data["grid"].shape[:2]
                    )
            origin = tuple(data["origin"])

            instance = cls(
                resolution=resolution,
                side_length=side_length,
                num_layers=num_layers,
                shape=shape,
                origin=origin,
                )
            instance._grid = np.asarray(data["grid"], dtype=np.float32)
        return instance

    # ── 清理 ──────────────────────────────────────────────────────────

    def reset(self):
        """重置地图为全未知 (-1)。"""
        self._grid.fill(-1.0)

    def clear_layer(self, layer: int = 0):
        """清空指定图层。"""
        self._grid[:, :, layer] = -1.0

    def __repr__(self) -> str:
        w, h, l = self._grid.shape
        return (
            f"Map2D({w}×{h}×{l} grid, "
            f"resolution={self.resolution}m, "
            f"size={self.side_length}m)"
            )
