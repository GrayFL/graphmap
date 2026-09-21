"""
SensorBridge — 原始传感器数据到 graphmap 的桥接层

核心功能:
    1. 内外参管理 (camera intrinsic, extrinsic body→camera, extrinsic body→lidar)
    2. LiDAR 点云 → 世界坐标系变换
    3. 3D 点 → 图像投影 & 可见性判断
    4. RGB 颜色映射到 LiDAR 点云
    5. 完整观测预处理, 对接 Environment.observe()

坐标系约定 (graphmap 统一):
    - 世界坐标系: Z 轴朝上的右手系
    - 机器人/相机自身坐标系: X=forward, Y=left, Z=up
    - 图像坐标系: h(高度/row), w(宽度/col), d(深度) 对应 x, y, z
    - 传感器内参 ppx/ppy/fx/fy → ppw/pph/fw/fh (且 f 取负, 因 left=-right, up=-down)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from numbers import Integral
from dataclasses import dataclass

from .pose import Pose
from .visibility import frustum_culling
from typing import Union, Literal
from .misc.config import Munch, load_config
from scipy import ndimage
from scipy.spatial.ckdtree import cKDTree

# ---------------------------------------------------------------------------
# CameraIntrinsic
# ---------------------------------------------------------------------------


@dataclass
class CameraIntrinsic:
    """相机内参封装，使用 graphmap 相机坐标约定。

    相机坐标系
    ----------
    X : forward
    Y : left
    Z : up

    图像坐标
    ----------
    h : height / row
    w : width / column

    投影关系
    ----------
    h = pph + fh * Z / X
    w = ppw + fw * Y / X

    由于 graphmap 的 left/up 与 OpenCV 的 right/down 方向相反：

        fw = -fx
        fh = -fy
        ppw = ppx
        pph = ppy

    Attributes
    ----------
    width : int
        图像宽度。
    height : int
        图像高度。
    pph : float
        垂直方向主点，等价于 OpenCV 的 ppy。
    ppw : float
        水平方向主点，等价于 OpenCV 的 ppx。
    fh : float
        graphmap 垂直焦距，通常为负数。
    fw : float
        graphmap 水平焦距，通常为负数。
    hfov : float
        水平视场角，单位为弧度。
    vfov : float
        垂直视场角，单位为弧度。
    """

    width: int
    height: int

    pph: float
    ppw: float

    fh: float
    fw: float

    hfov: float
    vfov: float

    def __post_init__(self) -> None:
        self.width = self._normalize_explicit_size(self.width, "width")
        self.height = self._normalize_explicit_size(self.height, "height")

        for name in ("pph", "ppw", "fh", "fw", "hfov", "vfov"):
            value = float(getattr(self, name))

            if not np.isfinite(value):
                raise ValueError(f"{name} 必须为有限数值，收到 {value!r}")

            setattr(self, name, value)

        if self.fh == 0.0 or self.fw == 0.0:
            raise ValueError("fh 和 fw 不能为 0")

        if not 0.0 < self.hfov < np.pi:
            raise ValueError(f"hfov 必须位于 (0, pi) 弧度内，收到 {self.hfov}")

        if not 0.0 < self.vfov < np.pi:
            raise ValueError(f"vfov 必须位于 (0, pi) 弧度内，收到 {self.vfov}")

    # ------------------------------------------------------------------
    # 焦距、FOV、图像尺寸换算
    # ------------------------------------------------------------------

    @staticmethod
    def focal_from_fov(size: float, fov: float) -> float:
        """根据图像尺寸和视场角计算焦距。

        公式
        ----
        focal = size / (2 * tan(fov / 2))

        Parameters
        ----------
        size
            width 或 height。
        fov
            对应方向的视场角，单位为弧度。

        Returns
        -------
        float
            OpenCV 约定下的正焦距。
        """
        size = float(size)
        fov = float(fov)

        if not np.isfinite(size) or size <= 0.0:
            raise ValueError(f"size 必须为正数，收到 {size!r}")

        if not np.isfinite(fov) or not 0.0 < fov < np.pi:
            raise ValueError(f"fov 必须位于 (0, pi) 弧度内，收到 {fov!r}")

        return size / (2.0 * np.tan(fov / 2.0))

    @staticmethod
    def fov_from_focal(size: float, focal: float) -> float:
        """根据图像尺寸和焦距计算视场角。

        公式
        ----
        fov = 2 * atan(size / (2 * focal))

        Parameters
        ----------
        size
            width 或 height。
        focal
            OpenCV 约定下的正焦距。

        Returns
        -------
        float
            视场角，单位为弧度。
        """
        size = float(size)
        focal = float(focal)

        if not np.isfinite(size) or size <= 0.0:
            raise ValueError(f"size 必须为正数，收到 {size!r}")

        if not np.isfinite(focal) or focal <= 0.0:
            raise ValueError(f"focal 必须为正数，收到 {focal!r}")

        return 2.0 * np.arctan(size / (2.0*focal))

    @staticmethod
    def size_from_focal_fov(focal: float, fov: float) -> float:
        """根据焦距和视场角计算图像尺寸。

        公式
        ----
        size = 2 * focal * tan(fov / 2)

        返回浮点数；用于图像尺寸时通常需要取整。
        """
        focal = float(focal)
        fov = float(fov)

        if not np.isfinite(focal) or focal <= 0.0:
            raise ValueError(f"focal 必须为正数，收到 {focal!r}")

        if not np.isfinite(fov) or not 0.0 < fov < np.pi:
            raise ValueError(f"fov 必须位于 (0, pi) 弧度内，收到 {fov!r}")

        return 2.0 * focal * np.tan(fov / 2.0)

    @staticmethod
    def graphmap_focal_from_sensor(focal: float) -> float:
        """将 OpenCV 正焦距转换为 graphmap 焦距。"""
        focal = float(focal)

        if not np.isfinite(focal) or focal <= 0.0:
            raise ValueError(f"传感器焦距必须为正数，收到 {focal!r}")

        return -focal

    @staticmethod
    def sensor_focal_from_graphmap(focal: float) -> float:
        """将 graphmap 焦距转换为 OpenCV 正焦距。"""
        focal = float(focal)

        if not np.isfinite(focal) or focal >= 0.0:
            raise ValueError(f"graphmap 焦距应为负数，收到 {focal!r}")

        return -focal

    # ------------------------------------------------------------------
    # 参数解析
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_explicit_size(value: float, name: str) -> int:
        """校验用户直接传入的 width 或 height。"""
        value = float(value)

        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须为正数，收到 {value!r}")

        rounded = int(round(value))

        if not np.isclose(value, rounded, rtol=0.0, atol=1e-6):
            raise ValueError(f"{name} 必须为整数，收到 {value!r}")

        return rounded

    @staticmethod
    def _normalize_derived_size(value: float, name: str) -> int:
        """将焦距和 FOV 推导出的图像尺寸转换为整数。"""
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"无法得到合法的 {name}: {value!r}")

        return max(1, int(round(value)))

    @classmethod
    def _resolve_axis(
        cls,
        *,
        axis_name: str,
        size: Optional[float],
        sensor_focal: Optional[float],
        graphmap_focal: Optional[float],
        sensor_pp: Optional[float],
        graphmap_pp: Optional[float],
        fov: Optional[float],
        ) -> tuple[int, float, float, float]:
        """解析一个方向上的尺寸、焦距、主点和 FOV。

        焦距优先级
        ----------
        1. sensor_focal，即 fx 或 fy
        2. graphmap_focal，即 fw 或 fh
        3. size + fov

        主点优先级
        ----------
        1. sensor_pp，即 ppx 或 ppy
        2. graphmap_pp，即 ppw 或 pph
        3. size / 2
        """
        size_i: Optional[int]

        if size is None:
            size_i = None
        else:
            size_i = cls._normalize_explicit_size(size, axis_name)

        # 优先使用 OpenCV 原始焦距。
        resolved_sensor_focal: Optional[float] = None

        if sensor_focal is not None:
            resolved_sensor_focal = float(sensor_focal)

            if (not np.isfinite(resolved_sensor_focal)
                    or resolved_sensor_focal <= 0.0):
                raise ValueError(
                    f"{axis_name} 方向的传感器焦距必须为正数，"
                    f"收到 {sensor_focal!r}"
                    )

        # 其次使用 graphmap 焦距。
        elif graphmap_focal is not None:
            resolved_sensor_focal = cls.sensor_focal_from_graphmap(
                graphmap_focal
                )

        # 最后尝试通过图像尺寸和 FOV 计算焦距。
        elif size_i is not None and fov is not None:
            resolved_sensor_focal = cls.focal_from_fov(size_i, fov)

        # 没有尺寸，但存在焦距和 FOV 时，反推出尺寸。
        if (size_i is None and resolved_sensor_focal is not None
                and fov is not None):
            derived_size = cls.size_from_focal_fov(
                resolved_sensor_focal,
                fov,
                )
            size_i = cls._normalize_derived_size(
                derived_size,
                axis_name,
                )

        if resolved_sensor_focal is None:
            raise ValueError(
                f"无法确定 {axis_name} 方向焦距。请提供以下任意一种组合："
                f"传感器焦距、graphmap 焦距，或 {axis_name} + FOV"
                )

        if size_i is None:
            raise ValueError(
                f"无法确定 {axis_name}。请直接提供 {axis_name}，"
                f"或同时提供焦距和对应 FOV"
                )

        # 主点优先使用传感器参数，其次使用 graphmap 参数。
        if sensor_pp is not None:
            resolved_pp = float(sensor_pp)
        elif graphmap_pp is not None:
            resolved_pp = float(graphmap_pp)
        else:
            # 没有提供主点时，默认位于图像几何中心。
            resolved_pp = size_i / 2.0

        if not np.isfinite(resolved_pp):
            raise ValueError(
                f"{axis_name} 方向主点必须为有限数值，"
                f"收到 {resolved_pp!r}"
                )

        resolved_graphmap_focal = cls.graphmap_focal_from_sensor(
            resolved_sensor_focal
            )

        # 以最终确定的 size 和 focal 为准重新计算 FOV。
        resolved_fov = cls.fov_from_focal(
            size_i,
            resolved_sensor_focal,
            )

        return (
            size_i,
            resolved_graphmap_focal,
            resolved_pp,
            resolved_fov,
            )

    @classmethod
    def from_sensor(
        cls,
        *,
        width: Optional[float] = None,
        height: Optional[float] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        ppx: Optional[float] = None,
        ppy: Optional[float] = None,
        ppw: Optional[float] = None,
        pph: Optional[float] = None,
        fw: Optional[float] = None,
        fh: Optional[float] = None,
        hfov: Optional[float] = None,
        vfov: Optional[float] = None,
        ) -> "CameraIntrinsic":
        """根据传感器参数或 graphmap 参数创建内参。

        支持的典型组合
        ----------------
        1. width, height, fx, fy, ppx, ppy
        2. width, height, fx, fy
        3. width, height, hfov, vfov
        4. width, height, fw, fh, ppw, pph
        5. fx, fy, hfov, vfov

        参数优先级
        ----------
        水平焦距:
            fx > fw > width + hfov

        垂直焦距:
            fy > fh > height + vfov

        水平主点:
            ppx > ppw > width / 2

        垂直主点:
            ppy > pph > height / 2

        说明
        ----
        当同时提供 width、fx 和 hfov 时，以 width 和 fx 为准，
        hfov 会根据最终参数重新计算。

        hfov 和 vfov 的单位均为弧度。
        """
        (
            resolved_width,
            resolved_fw,
            resolved_ppw,
            resolved_hfov,
            ) = cls._resolve_axis(
                axis_name="width",
                size=width,
                sensor_focal=fx,
                graphmap_focal=fw,
                sensor_pp=ppx,
                graphmap_pp=ppw,
                fov=hfov,
                )

        (
            resolved_height,
            resolved_fh,
            resolved_pph,
            resolved_vfov,
            ) = cls._resolve_axis(
                axis_name="height",
                size=height,
                sensor_focal=fy,
                graphmap_focal=fh,
                sensor_pp=ppy,
                graphmap_pp=pph,
                fov=vfov,
                )

        return cls(
            width=resolved_width,
            height=resolved_height,
            pph=resolved_pph,
            ppw=resolved_ppw,
            fh=resolved_fh,
            fw=resolved_fw,
            hfov=resolved_hfov,
            vfov=resolved_vfov,
            )

    # ------------------------------------------------------------------
    # OpenCV 参数访问
    # ------------------------------------------------------------------

    @property
    def fx(self) -> float:
        """OpenCV 水平方向正焦距。"""
        return -self.fw

    @property
    def fy(self) -> float:
        """OpenCV 垂直方向正焦距。"""
        return -self.fh

    @property
    def ppx(self) -> float:
        """OpenCV 水平方向主点。"""
        return self.ppw

    @property
    def ppy(self) -> float:
        """OpenCV 垂直方向主点。"""
        return self.pph

    # ------------------------------------------------------------------
    # 投影和反投影
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """返回直接作用于 graphmap 相机坐标的 3×3 投影矩阵。

        输入点为：

            [X, Y, Z]

        变换后：

            [h']   [pph,  0, fh] [X]
            [w'] = [ppw, fw,  0] [Y]
            [X']   [  1,  0,  0] [Z]

        归一化后：

            h = h' / X' = pph + fh * Z / X
            w = w' / X' = ppw + fw * Y / X
        """
        return np.array(
            [
                [self.pph, 0.0, self.fh],
                [self.ppw, self.fw, 0.0],
                [1.0, 0.0, 0.0],
                ],
            dtype=np.float32,
            )

    def cam_to_hwd(self, pts_camera: np.ndarray) -> np.ndarray:
        """将 graphmap 相机坐标转换为 hwd 图像坐标。

        Parameters
        ----------
        pts_camera
            shape 为 (N, 3)，每行为 (X, Y, Z)。

        Returns
        -------
        np.ndarray
            shape 为 (N, 3)，每行为 (h, w, d)。

            d 轴朝向观察者，因此相机前方的点满足 d < 0。
        """
        pts_camera = np.asarray(pts_camera)

        if pts_camera.ndim != 2 or pts_camera.shape[1] != 3:
            raise ValueError("pts_camera 必须为 shape (N, 3)")

        res = pts_camera @ self.matrix.T

        physical_depth = res[:, 2:3]

        # 仅在除法时替换零深度，不修改最终返回的真实深度。
        safe_depth = np.where(
            physical_depth == 0.0,
            1e-8,
            physical_depth,
            )

        hw = res[:, :2] / safe_depth

        # hwd 中 d 轴朝向观察者。
        d = -physical_depth

        return np.concatenate([hw, d], axis=-1)

    def hwd_to_cam(self, pts_hwd: np.ndarray) -> np.ndarray:
        """将 hwd 图像坐标反变换为 graphmap 相机坐标。

        Parameters
        ----------
        pts_hwd
            shape 为 (N, 3)，每行为 (h, w, d)。

        Returns
        -------
        np.ndarray
            shape 为 (N, 3)，每行为 (X, Y, Z)。
        """
        pts_hwd = np.asarray(pts_hwd)

        if pts_hwd.ndim != 2 or pts_hwd.shape[1] != 3:
            raise ValueError("pts_hwd 必须为 shape (N, 3)")

        h = pts_hwd[:, 0:1]
        w = pts_hwd[:, 1:2]
        d = pts_hwd[:, 2:3]

        # d 轴向外，因此前向物理深度 X = -d。
        X = -d

        # h = pph + fh * Z / X
        Z = (h - self.pph) * X / self.fh

        # w = ppw + fw * Y / X
        Y = (w - self.ppw) * X / self.fw

        return np.concatenate([X, Y, Z], axis=-1)

    # ------------------------------------------------------------------
    # 图像网格采样
    # ------------------------------------------------------------------

    def create_grid(
            self,
            stride: Union[int, tuple[int, int]] = 1,
            order: str = "C"
        ) -> np.ndarray:
        """创建以光心为锚点的图像采样网格。

        Parameters
        ----------
        stride
            采样步长。
            可以传入单个整数：
                stride=8
            表示高、宽方向均使用步长 8。
            也可以传入：
                stride=(stride_h, stride_w)

        Returns
        -------
        np.ndarray
            shape 为 (N, 2)，每行为整数图像索引：
                [h, w]
            所有坐标均满足：
                0 <= h < height
                0 <= w < width
            返回顺序按照采样点到实际光心 (pph, ppw) 的距离，
            由近到远排列。

        Notes
        -----
        光心可能为小数，而图像索引必须为整数，因此首先将光心
        四舍五入到最近的整数像素，并以该像素作为网格锚点。
        """
        if isinstance(stride, Integral):
            stride_h = int(stride)
            stride_w = int(stride)
        else:
            if len(stride) != 2:
                raise ValueError("stride 元组必须为 (stride_h, stride_w)")

            stride_h, stride_w = map(int, stride)

        if stride_h <= 0 or stride_w <= 0:
            raise ValueError("stride 必须为正整数")

        # 不使用 np.rint，避免其银行家舍入行为。
        center_h = int(np.floor(self.pph + 0.5))
        center_w = int(np.floor(self.ppw + 0.5))

        # 通过取模找到图像范围内与光心锚点同相位的第一个坐标。
        start_h = center_h % stride_h
        start_w = center_w % stride_w

        h_coordinates = np.arange(
            start_h, self.height, stride_h, dtype=np.int64
            )
        w_coordinates = np.arange(
            start_w, self.width, stride_w, dtype=np.int64
            )
        h_grid, w_grid = np.meshgrid(
            h_coordinates,
            w_coordinates,
            indexing="ij"
        )
        grid = np.stack([h_grid.ravel(order='C'), w_grid.ravel(order='C')],
                        axis=-1)

        normalized_order = str(order).lower()
        if normalized_order in {"c", "raster", "row-major", "row_major"}:
            return grid
        if normalized_order in {"center", "distance"}:
            # 按照距离真实浮点光心的距离排序。
            distance_squared = ((grid[:, 0] - self.pph)**2 +
                                (grid[:, 1] - self.ppw)**2)
            sort_index = np.argsort(distance_squared, kind="stable")
            return grid[sort_index]
        raise ValueError("order 必须为 'C'（左上到右下）或 'center'（按光心距离）")


# ---------------------------------------------------------------------------
# SensorBridge
# ---------------------------------------------------------------------------


class SensorBridge:
    """传感器桥接器: 将原始 LiDAR + RGB + TF 数据处理为 graphmap 可用的观测

    Usage:
        bridge = SensorBridge(
            intrinsic=CameraIntrinsic.from_sensor(fx=606.8, fy=605.6, ppx=324.7, ppy=241.3),
            extrinsic_body_to_camera=ext_body2cam,
            extrinsic_body_to_lidar=ext_body2lidar,
        )

        # 方式 1: 手动分步处理
        pts_world, colors = bridge.colorize_lidar(
            pts_lidar, lidar_pose, camera_pose, rgb_image
        )
        pts_camera = bridge.world_to_camera(pts_world, camera_pose)
        env.observe(pose_robot=camera_pose, points_camera=pts_camera, colors=colors)

        # 方式 2: 一键处理观测 JSON
        result = bridge.process_observation(obs_json_dict)
        env.observe(**result)
    """

    def __init__(
        self,
        config: Munch,
        intrinsic_camera: CameraIntrinsic,
        extrinsic_body_to_camera: Optional[Pose] = None,
        extrinsic_body_to_lidar3d: Optional[Pose] = None,
        depth_range: Tuple[float, float] = (0.1, 100.0)
        ):
        """
        Parameters
        ----------
        intrinsic : CameraIntrinsic
            相机内参
        extrinsic_body_to_camera : np.ndarray, optional
            (4, 4) 齐次变换矩阵: body → camera, 默认单位矩阵
        extrinsic_body_to_lidar : np.ndarray, optional
            (4, 4) 齐次变换矩阵: body → LiDAR, 默认单位矩阵
        depth_range : Tuple[float, float]
            (min_depth, max_depth), 深度裁剪范围
        """
        self.config = config
        self.intrinsic = intrinsic_camera
        self.ext_body2camera = extrinsic_body_to_camera
        self.ext_body2lidar3d = extrinsic_body_to_lidar3d
        self.depth_min, self.depth_max = depth_range

    def colorize_points(
        self,
        pts_lidar: np.ndarray,
        sensor_pose: Pose,
        rgb_image: np.ndarray,
        intrinsic: CameraIntrinsic,
        default_color: Tuple[int, int, int] = (128, 128, 128)
        ) -> Tuple[np.ndarray, np.ndarray]:
        """将 LiDAR 点云变换到世界系并从 RGB 图像取色

        流程:
            1. LiDAR 点 → 世界系
            2. 世界点 → 相机系
            3. 相机系点 → 图像投影
            4. 从 RGB 图像采样颜色

        Parameters
        ----------
        pts_lidar : np.ndarray
            shape (N, 3) float32, LiDAR 坐标系点云
        sensor_pose : Pose
            传感器相对相机系位姿，例如 cam2lidar
        rgb_image : np.ndarray
            shape (H, W, 3) uint8, RGB 图像
        default_color : Tuple[int, int, int]
            不可见点的默认颜色

        Returns
        -------
        pts_world : np.ndarray
            shape (N, 3) float32, 世界系点云
        colors : np.ndarray
            shape (N, 3) or (N, 4) uint8, RGBA 颜色
        """
        N = pts_lidar.shape[0]
        img_h, img_w = rgb_image.shape[:2]

        pts_cam = sensor_pose(pts_lidar)
        pts_hwd = intrinsic.cam_to_hwd(pts_cam)
        h, w, d = pts_hwd[:, 0], pts_hwd[:, 1], pts_hwd[:, 2]

        # 4. 构建可见性 Mask (深度有效 且 在图像范围内)
        # 注意：HWD系统中，在相机前方的物体，其 d 是负数，物理深度为 -d
        valid_depth = (d < -self.depth_min)
        in_bounds = (h >= 0) & (h < img_h) & (w >= 0) & (w < img_w)
        visible_mask = valid_depth & in_bounds

        pts_hwd = pts_hwd[visible_mask]

        # 4. 采样颜色
        colors = np.full((N, 3), default_color, dtype=np.uint8)
        if np.any(visible_mask):
            vis_h = np.clip(h[visible_mask].astype(np.int32), 0, img_h - 1)
            vis_w = np.clip(w[visible_mask].astype(np.int32), 0, img_w - 1)

            # 从 RGB 图像中高级索引取色
            colors[visible_mask] = rgb_image[vis_h, vis_w]
        return Munch(
            visible_mask=visible_mask,
            colors=colors,
            pts_hwd=pts_hwd,
            valid_depth=valid_depth,
            in_bounds=in_bounds,
            )

    def sample_points_from_hw_data(
        self,
        points: np.ndarray,
        T_pts2cam: Pose,
        hw_data: np.ndarray,
        *,
        intrinsic: CameraIntrinsic = None,
        default_value: Any = 0,
        index_mode: Literal["floor", "nearest"] = "nearest",
        ) -> Munch:
        """
        将任意坐标系下的一组空间点投影到图像，并从 H-W 数据中采样。

        Parameters
        ----------
        points : np.ndarray
            shape (N, 3)，某个源坐标系下的点。
        T_pts2cam : Pose
            必须满足：
                pts_cam = T_pts2cam(points)
            也就是说，它需要把 points 所在坐标系的点变换到相机坐标系。
        hw_data : np.ndarray
            shape (H, W) 或 (H, W, C...)。
            可以是 RGB 图、深度图、语义图、特征图等。
        intrinsic : CameraIntrinsic
            相机内参，提供 cam_to_hwd。
        default_value : Any
            不可见点的默认采样值。
        index_mode : {"floor", "nearest"}
            图像坐标到像素索引的转换方式。

        Returns
        -------
        Munch
            samples : np.ndarray
                shape (N,) 或 (N, C...)。
            visible_mask : np.ndarray
                shape (N,)。
            pts_cam : np.ndarray
                shape (N, 3)。
            pts_hwd : np.ndarray
                shape (N, 3)，完整的投影结果。
            visible_pts_hwd : np.ndarray
                shape (M, 3)，只包含可见点。
            valid_depth : np.ndarray
                shape (N,)。
            in_bounds : np.ndarray
                shape (N,)。
        """
        points = np.asarray(points)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points 必须是 shape (N, 3)，当前是 {points.shape}")

        pts_cam = T_pts2cam(points)
        intrinsic = intrinsic or self.intrinsic
        pts_hwd = intrinsic.cam_to_hwd(pts_cam)

        h = pts_hwd[:, 0]
        w = pts_hwd[:, 1]
        d = pts_hwd[:, 2]

        # 你的 HWD 约定：相机前方 d 为负数，物理深度为 -d
        valid_depth = d < -self.depth_min

        sampled = sample_hw_data(
            data=hw_data,
            h=h,
            w=w,
            valid_mask=valid_depth,
            default_value=default_value,
            index_mode=index_mode,
            )

        return Munch(
            samples=sampled.samples,
            visible_mask=sampled.visible_mask,
            pts_cam=pts_cam,
            pts_hwd=pts_hwd,
            visible_pts_hwd=pts_hwd[sampled.visible_mask],
            valid_depth=valid_depth,
            in_bounds=sampled.in_bounds,
            index_h=sampled.index_h,
            index_w=sampled.index_w,
            )

    # ------------------------------------------------------------------
    # 完整观测处理
    # ------------------------------------------------------------------

    def process_observation(
            self,
            obs_data: Dict[str, Any],
            pose_body: Optional[Pose] = None,
            rgb_image: Optional[np.ndarray] = None,
            pts_lidar: Optional[np.ndarray] = None
        ) -> Dict[str, Any]:
        """处理一次完整观测, 返回可直接传入 env.observe() 的参数字典

        支持两种输入方式:
        1. 从 JSON 字典自动提取: obs_data 需包含 'tf', 'camera', 'lidar3d' 键
        2. 显式传入: body_pose, rgb_image, pts_lidar (此时 obs_data 仅用 'tf')

        Parameters
        ----------
        obs_data : dict
            观测数据, 格式:
            {
                'tf': {'ret': [position (3,), quaternion (4,)]},
                'camera': {'ret': [[[r,g,b],...],...]},  # (H,W,3) nested list
                'lidar3d': {'ret': [[x,y,z],...]},       # (N,3) nested list
            }
        body_pose : Pose, optional
            机器人 body 在世界系的位姿 (从 tf 计算), 不传则从 obs_data['tf'] 构造
        rgb_image : np.ndarray, optional
            (H, W, 3) uint8 图像, 不传则从 obs_data['camera']['ret'] 构造
        pts_lidar : np.ndarray, optional
            (N, 3) float32, LiDAR 点云, 不传则从 obs_data['lidar3d']['ret'] 构造

        Returns
        -------
        dict
            {'pose_robot': Pose, 'points_camera': np.ndarray, 'colors': np.ndarray}
            可直接 ** 解包传入 env.observe()
        """
        timestamp = obs_data['tf']['timestamp']['data']

        # --- 解析 body 位姿 ---
        if pose_body is None:
            tf_data = obs_data['tf']['ret']
            pose_body = Pose(
                position=tf_data[0],
                rotation=tf_data[1],  # quaternion [x,y,z,w]
                label="body in world",
                )

        # --- 计算传感器位姿 ---
        pose_camera = pose_body * self.ext_body2camera
        pose_lidar3d = pose_body * self.ext_body2lidar3d

        # --- 解析 LiDAR 点云 ---
        if pts_lidar is None:
            pts_lidar = np.array(
                obs_data['lidar3d']['ret'], dtype=np.float32
                )

        pts_world = pose_lidar3d(pts_lidar)

        mask_floor = (
            (pts_world[:, 2] < self.config.map.clip_z[0])
            & (pts_world[:, 2] > self.config.map.under_floor_height)
            )
        pts_floor = pts_world[mask_floor]

        # mask_map_build = (
        #     (pts_world[:, 2] > self.config.map.clip_z[0])
        #     & (pts_world[:, 2] < self.config.map.clip_z[1])
        #     )  #
        # pts_world = pts_world[mask_map_build]
        # pts_lidar = pts_lidar[mask_map_build]

        # --- 解析 RGB 图像 ---
        if rgb_image is None:
            rgb_image = np.array(obs_data['camera']['ret'], dtype=np.uint8)

        # --- 颜色映射 & 世界变换 ---
        # cam2lidar = (~pose_camera) * pose_lidar3d
        cam2world = ~pose_camera
        ret = self.colorize_points(
            pts_world, cam2world, rgb_image, self.intrinsic
            )
        return {
            'pose_robot': pose_body,
            'pose_camera': pose_camera,
            'pose_lidar3d': pose_lidar3d,
            'points_world': pts_world,
            'points_floor': pts_floor,
            'points_hwd': ret.pts_hwd,
            'visible_mask': ret.visible_mask,
            'colors': ret.colors,
            'timestamp': timestamp,
            }

    def process_observation_world(
            self,
            pose_body: "Pose",
            pts_world: np.ndarray,
            rgb_image: np.ndarray,
            timestamp: Optional[float] = None
        ) -> Dict[str, Any]:
        """处理一次完整观测（世界坐标系点云），返回可直接传入 env.observe() 的参数字典。

        与 ``process_observation`` 的区别：
        - 点云已处于世界坐标系，跳过 lidar→world 变换。
        - 跳过雷达自身系的 frustum culling（点云已由外部注册系统处理）。

        Parameters
        ----------
        pose_body : Pose
            机器人 body 在世界系的位姿（来自 odometry）。
        pts_world : np.ndarray
            (N, 3) float32，世界坐标系中的 LiDAR 点云。
        rgb_image : np.ndarray
            (H, W, 3) uint8 图像。
        timestamp : float, optional
            观测时间戳（秒）。

        Returns
        -------
        dict
            与 ``process_observation`` 相同格式，可直接 ** 解包传入 env.observe()。
        """
        # --- 计算传感器位姿 ---
        pose_camera = pose_body * self.ext_body2camera
        pose_lidar3d = pose_body * self.ext_body2lidar3d

        pts_world = np.asarray(pts_world, dtype=np.float32)

        # --- 地面过滤 & Z 轴裁剪（与旧方法一致） ---
        mask_floor = (
            (pts_world[:, 2] < self.config.map.clip_z[0])
            & (pts_world[:, 2] > self.config.map.under_floor_height)
            )
        pts_floor = pts_world[mask_floor]

        mask_map_build = ((pts_world[:, 2] > self.config.map.clip_z[0])
                            &
                            (pts_world[:, 2] < self.config.map.clip_z[1]))
        pts_world = pts_world[mask_map_build]

        # --- 颜色映射 ---
        cam2world = ~pose_camera
        ret = self.colorize_points(
            pts_world, cam2world, rgb_image, self.intrinsic
            )
        return {
            'pose_robot': pose_body,
            'pose_camera': pose_camera,
            'pose_lidar3d': pose_lidar3d,
            'points_world': pts_world,
            'points_floor': pts_floor,
            'points_hwd': ret.pts_hwd,
            'visible_mask': ret.visible_mask,
            'colors': ret.colors,
            'timestamp': timestamp,
            }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def filter_by_frustum(
            self,
            pts_world: np.ndarray,
            camera_pose: Pose,
            fov_h=None,
            fov_v=None,
            near: float = 0.1,
            far: float = 100.0
        ) -> np.ndarray:
        """视锥体裁剪: 返回在世界系视锥内的点索引掩码"""
        return frustum_culling(
            pts_world, camera_pose, fov_h, fov_v, near, far
            )

    @classmethod
    def from_rsim_config(
            cls, config: Union[str, Dict, Munch]
        ) -> "SensorBridge":
        """从 rsim 兼容的 config (Munch / dict) 构造 SensorBridge

        自动映射:
            config.camera_params.rgb_sensor.fx/fy/ppx/ppy → CameraIntrinsic.from_sensor()
            config.camera_params.rgb_sensor.position/rotation → ext_body2cam
            config.camera_params.lidar3d_sensor.position/rotation → ext_body2lidar

        rotation 为 euler 角度 (degrees, xyz 顺序), 表示传感器在 body 系下的朝向.
        """
        from scipy.spatial.transform import Rotation as R

        if isinstance(config, str):
            config = load_config(config)
        elif isinstance(config, dict):
            config = Munch.fromDict(config)

        # 兼容 camera_params 和 sensor_params 两种 key
        cfg = config.get('camera_params'
                        ) or config.get('sensor_params') or config
        cfg_rgb = cfg.rgb_sensor
        cfg_lid = cfg.lidar3d_sensor

        intrinsic_camera = CameraIntrinsic.from_sensor(
            fx=cfg_rgb.fx,
            fy=cfg_rgb.fy,
            ppx=cfg_rgb.ppx,
            ppy=cfg_rgb.ppy,
            )

        ext_body2cam = Pose(
            position=np.array(cfg_rgb.position, dtype=np.float32),
            rotation=R.from_euler('xyz', cfg_rgb.rotation, degrees=True),
            label="body->camera"
            )
        ext_body2lid3d = Pose(
            position=np.array(cfg_lid.position, dtype=np.float32),
            rotation=R.from_euler('xyz', cfg_lid.rotation, degrees=True),
            label="body->lidar3d"
            )

        return cls(
            config=config,
            intrinsic_camera=intrinsic_camera,
            extrinsic_body_to_camera=ext_body2cam,
            extrinsic_body_to_lidar3d=ext_body2lid3d
            )


def load_observation_from_json(path: str) -> Dict[str, Any]:
    """从 JSON 文件加载观测数据 (仅加载, 不解析数组)"""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class RayDirLUTEstimator:
    """
    从多帧无序 lidar 点云估计扁平射线方向表。

    输出:
        ray_dirs_lidar: shape = (K, 3)

    特点:
        1. 不需要 H/W
        2. 不需要 FOV
        3. 不使用均匀初始化
        4. 先角度哈希，再做近邻合并
    """

    def __init__(
            self,
            min_range=0.5,
            max_range=100.0,
            cell_deg=None,
            merge_deg=None,
            assign_deg=None,
            min_count=2,
            max_angular_std_deg=0.16,
            auto_sample_size=200_000,
            dtype=np.float32
        ):
        self.min_range = float(min_range)
        self.max_range = float(max_range)

        self.cell_deg = cell_deg
        self.merge_deg = merge_deg
        self.assign_deg = assign_deg

        self.min_count = int(min_count)
        self.max_angular_std = np.deg2rad(max_angular_std_deg)

        self.auto_sample_size = int(auto_sample_size)
        self.dtype = dtype

    @staticmethod
    def _normalize(v):
        return v / np.maximum(
            np.linalg.norm(v, axis=1, keepdims=True), 1e-12
            )

    def _points_to_dirs(self, pts_lidar):
        pts = np.asarray(pts_lidar, dtype=np.float64)
        r = np.linalg.norm(pts, axis=1)

        valid = (
            np.isfinite(r)
            & (r >= self.min_range)
            & (r <= self.max_range)
            & (pts[:, 0] > 1e-6)
            )

        pts = pts[valid]
        r = r[valid]

        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)

        return pts / np.maximum(r[:, None], 1e-12)

    def _collect_dirs(self, frames_lidar):
        dirs_all = []

        for pts in frames_lidar:
            d = self._points_to_dirs(pts)
            if len(d) > 0:
                dirs_all.append(d)

        if len(dirs_all) == 0:
            return np.empty((0, 3), dtype=np.float64)

        return np.concatenate(dirs_all, axis=0)

    @staticmethod
    def _az_el_from_dirs(dirs):
        x = dirs[:, 0]
        y = dirs[:, 1]
        z = dirs[:, 2]

        az = np.arctan2(y, x)
        el = np.arctan2(z, np.sqrt(x*x + y*y))

        return az, el

    def _auto_estimate_scales(self, dirs):
        """
        自动估计 cell_deg / merge_deg / assign_deg。

        注意：
            这里用 sample 的最近邻角度估计。
            如果 sample 中包含大量同一射线的重复观测，则最近邻可能偏小。
            因此只作为默认值，工程上建议你根据日志微调。
        """
        n = len(dirs)
        if n == 0:
            raise ValueError("No valid lidar directions.")

        if n > self.auto_sample_size:
            idx = np.random.default_rng(0).choice(
                n, self.auto_sample_size, replace=False
                )
            sample = dirs[idx]
        else:
            sample = dirs

        tree = cKDTree(sample)
        dist, _ = tree.query(sample, k=2, workers=-1)

        nn_chord = dist[:, 1]
        nn_ang = 2.0 * np.arcsin(np.clip(0.5 * nn_chord, 0.0, 1.0))
        nn_deg = np.rad2deg(nn_ang)

        # 过滤掉极端重复点 / 异常点
        p10 = np.percentile(nn_deg, 10)
        p50 = np.percentile(nn_deg, 50)

        # 防止重复观测导致 p10 过小
        base = max(p10, 0.25 * p50)

        cell_deg = self.cell_deg
        merge_deg = self.merge_deg
        assign_deg = self.assign_deg

        if cell_deg is None:
            cell_deg = max(0.25 * base, 0.005)

        if merge_deg is None:
            merge_deg = max(0.60 * base, 0.01)

        if assign_deg is None:
            assign_deg = max(0.80 * base, 0.015)

        stat = {
            "nn_deg_p10": p10,
            "nn_deg_p50": p50,
            "auto_cell_deg": cell_deg,
            "auto_merge_deg": merge_deg,
            "auto_assign_deg": assign_deg,
            }

        return cell_deg, merge_deg, assign_deg, stat

    def _coarse_hash_clusters(self, dirs, cell_deg):
        """
        使用 az/el 稀疏角度哈希进行第一阶段合并。
        """
        az, el = self._az_el_from_dirs(dirs)

        cell = np.deg2rad(cell_deg)

        iaz = np.floor(az / cell).astype(np.int64)
        iel = np.floor(el / cell).astype(np.int64)

        # 将二维整数 cell 编成一个 64-bit key
        # 加偏移避免负数带来的直观混乱
        offset = np.int64(1 << 30)
        key = (iaz+offset) * np.int64(2_000_000_003) + (iel+offset)

        uniq, inv = np.unique(key, return_inverse=True)
        K0 = len(uniq)

        sum_x = np.bincount(inv, weights=dirs[:, 0], minlength=K0)
        sum_y = np.bincount(inv, weights=dirs[:, 1], minlength=K0)
        sum_z = np.bincount(inv, weights=dirs[:, 2], minlength=K0)
        count = np.bincount(inv, minlength=K0).astype(np.int64)

        sums = np.stack([sum_x, sum_y, sum_z], axis=1)
        centers = self._normalize(sums)

        return centers, sums, count

    def _merge_close_centers(self, sums, count, merge_deg, max_rounds=3):
        """
        对 coarse centers 再做球面近邻合并。
        """
        merge_chord = 2.0 * np.sin(0.5 * np.deg2rad(merge_deg))

        for _ in range(max_rounds):
            centers = self._normalize(sums)
            K = len(centers)

            if K == 0:
                break

            tree = cKDTree(centers)
            pairs = tree.query_pairs(r=merge_chord)

            if len(pairs) == 0:
                break

            parent = np.arange(K, dtype=np.int64)

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra = find(a)
                rb = find(b)
                if ra == rb:
                    return
                if count[ra] >= count[rb]:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

            for a, b in pairs:
                union(a, b)

            roots = np.array([find(i) for i in range(K)], dtype=np.int64)
            _, inv = np.unique(roots, return_inverse=True)
            K2 = int(inv.max()) + 1

            new_sums = np.zeros((K2, 3), dtype=np.float64)
            new_count = np.zeros(K2, dtype=np.int64)

            np.add.at(new_sums, inv, sums)
            np.add.at(new_count, inv, count)

            if K2 == K:
                break

            sums = new_sums
            count = new_count

        centers = self._normalize(sums)
        return centers, sums, count

    def _reassign_to_centers(self, dirs, centers, assign_deg):
        """
        将原始方向重新分配到合并后的中心，得到更准确的均值和角度方差。
        """
        K = len(centers)
        if K == 0:
            return centers, np.zeros((0, 3)), np.zeros((0,), dtype=np.int64)

        assign_chord = 2.0 * np.sin(0.5 * np.deg2rad(assign_deg))

        tree = cKDTree(centers)
        dist, idx = tree.query(
            dirs,
            k=1,
            workers=-1,
            distance_upper_bound=assign_chord,
        )

        ok = idx < K

        idx_ok = idx[ok]
        dirs_ok = dirs[ok]

        sum_x = np.bincount(idx_ok, weights=dirs_ok[:, 0], minlength=K)
        sum_y = np.bincount(idx_ok, weights=dirs_ok[:, 1], minlength=K)
        sum_z = np.bincount(idx_ok, weights=dirs_ok[:, 2], minlength=K)
        count = np.bincount(idx_ok, minlength=K).astype(np.int64)

        sums = np.stack([sum_x, sum_y, sum_z], axis=1)

        valid = count > 0
        new_centers = centers.copy()
        new_centers[valid] = self._normalize(sums[valid])

        return new_centers, sums, count

    def fit(self, frames_lidar, verbose=False):
        dirs = self._collect_dirs(frames_lidar)

        if len(dirs) == 0:
            raise ValueError(
                "No valid lidar points after range filtering."
                )

        cell_deg, merge_deg, assign_deg, auto_stat = self._auto_estimate_scales(dirs)

        if verbose:
            print(f"valid direction samples: {len(dirs)}")
            print(
                "auto scale: "
                f"cell={cell_deg:.6f} deg, "
                f"merge={merge_deg:.6f} deg, "
                f"assign={assign_deg:.6f} deg"
                )
            print(
                "nn scale: "
                f"p10={auto_stat['nn_deg_p10']:.6f} deg, "
                f"p50={auto_stat['nn_deg_p50']:.6f} deg"
                )

        centers, sums, count = self._coarse_hash_clusters(dirs, cell_deg)

        if verbose:
            print(f"after angular hash: clusters={len(centers)}")

        centers, sums, count = self._merge_close_centers(
            sums=sums,
            count=count,
            merge_deg=merge_deg,
            max_rounds=3,
        )

        if verbose:
            print(f"after merge: clusters={len(centers)}")

        centers, sums, count = self._reassign_to_centers(
            dirs=dirs,
            centers=centers,
            assign_deg=assign_deg,
        )

        if verbose:
            print(f"after reassign: clusters={np.sum(count > 0)}")

        resultant = np.zeros(len(count), dtype=np.float64)
        has = count > 0
        resultant[has] = np.linalg.norm(
            sums[has], axis=1
            ) / np.maximum(count[has], 1)

        angular_std = np.full(len(count), np.inf, dtype=np.float64)
        angular_std[has] = np.sqrt(
            np.maximum(2.0 * (1.0 - resultant[has]), 0.0)
            )

        valid = ((count >= self.min_count)
                    & np.isfinite(angular_std)
                    & (angular_std <= self.max_angular_std))

        ray_dirs = centers[valid]
        ray_count = count[valid]
        ray_std = angular_std[valid]

        az, el = self._az_el_from_dirs(ray_dirs)
        order = np.lexsort((az, -el))

        ray_dirs = ray_dirs[order]
        ray_count = ray_count[order]
        ray_std = ray_std[order]
        az = az[order]
        el = el[order]

        stat = {
            **auto_stat,
            "cell_deg": cell_deg,
            "merge_deg": merge_deg,
            "assign_deg": assign_deg,
            "count": ray_count,
            "angular_std_rad": ray_std.astype(np.float32),
            "angular_std_deg": np.rad2deg(ray_std).astype(np.float32),
            "azimuth_rad": az.astype(np.float32),
            "elevation_rad": el.astype(np.float32),
            "azimuth_deg": np.rad2deg(az).astype(np.float32),
            "elevation_deg": np.rad2deg(el).astype(np.float32),
            "num_rays": len(ray_dirs),
            }

        return ray_dirs.astype(self.dtype), stat


def sample_hw_data(
    data: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    default_value: Any = 0,
    index_mode: Literal["floor", "nearest"] = "nearest",
    ) -> Munch:
    """
    从任意 H-W 数据中按照 h, w 采样。

    Parameters
    ----------
    data : np.ndarray
        shape (H, W) 或 (H, W, C...)。
        可以是 RGB、深度图、语义标签、feature map 等。
    h : np.ndarray
        shape (N,)，图像纵坐标。
    w : np.ndarray
        shape (N,)，图像横坐标。
    valid_mask : Optional[np.ndarray]
        shape (N,)，额外有效性 mask，例如深度有效 mask。
    default_value : Any
        不可见点的默认值。
        若 data 是 (H, W)，则 default_value 应为标量。
        若 data 是 (H, W, C)，则 default_value 可以是标量或 shape (C,)。
    index_mode : {"floor", "nearest"}
        float 坐标转像素索引的方式。
        "floor" 更接近你原来的 astype(np.int32) 行为。

    Returns
    -------
    Munch
        samples : np.ndarray
            shape (N,) 或 (N, C...)。
        visible_mask : np.ndarray
            shape (N,)，最终可采样 mask。
        in_bounds : np.ndarray
            shape (N,)，是否落在图像范围内。
        index_h : np.ndarray
            shape (M,)，可见点使用的 h 索引。
        index_w : np.ndarray
            shape (M,)，可见点使用的 w 索引。
    """
    data = np.asarray(data)

    if data.ndim < 2:
        raise ValueError(f"data 至少需要是二维数组，当前 data.ndim={data.ndim}")

    h = np.asarray(h)
    w = np.asarray(w)

    if h.shape != w.shape:
        raise ValueError(f"h 和 w 的 shape 必须一致，得到 h={h.shape}, w={w.shape}")

    if h.ndim != 1:
        raise ValueError(f"当前实现假设 h/w 是 shape (N,)；得到 h.shape={h.shape}")

    img_h, img_w = data.shape[:2]
    sample_shape = data.shape[2:]
    num_points = h.shape[0]

    default_arr = _make_default_value(
        default_value=default_value,
        sample_shape=sample_shape,
        dtype=data.dtype,
        )

    samples = np.empty((num_points, ) + sample_shape, dtype=data.dtype)
    samples[...] = default_arr

    finite_mask = np.isfinite(h) & np.isfinite(w)
    in_bounds = (
        finite_mask
        & (h >= 0)
        & (h < img_h)
        & (w >= 0)
        & (w < img_w)
        )

    if valid_mask is None:
        visible_mask = in_bounds
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != h.shape:
            raise ValueError(
                f"valid_mask.shape 必须等于 h.shape，"
                f"得到 valid_mask={valid_mask.shape}, h={h.shape}"
                )
        visible_mask = valid_mask & in_bounds

    if np.any(visible_mask):
        if index_mode == "floor":
            index_h = np.floor(h[visible_mask]).astype(np.int32)
            index_w = np.floor(w[visible_mask]).astype(np.int32)
        elif index_mode == "nearest":
            index_h = np.rint(h[visible_mask]).astype(np.int32)
            index_w = np.rint(w[visible_mask]).astype(np.int32)
        else:
            raise ValueError(f"未知 index_mode={index_mode}")

        index_h = np.clip(index_h, 0, img_h - 1)
        index_w = np.clip(index_w, 0, img_w - 1)

        samples[visible_mask] = data[index_h, index_w]
    else:
        index_h = np.empty((0, ), dtype=np.int32)
        index_w = np.empty((0, ), dtype=np.int32)

    return Munch(
        samples=samples,
        visible_mask=visible_mask,
        in_bounds=in_bounds,
        index_h=index_h,
        index_w=index_w,
        )


def _make_default_value(
    default_value: Any,
    sample_shape: tuple,
    dtype: np.dtype,
    ) -> np.ndarray:
    """
    将 default_value 整理成可赋值到单个采样结果的 shape。

    data shape:
        (H, W)       -> sample_shape = ()
        (H, W, C)    -> sample_shape = (C,)
        (H, W, C, D) -> sample_shape = (C, D)
    """
    default_arr = np.asarray(default_value, dtype=dtype)

    if sample_shape == ():
        if default_arr.shape == ():
            return default_arr
        if default_arr.size == 1:
            return default_arr.reshape(())
        raise ValueError(
            f"data 是二维的，因此 default_value 必须是标量；"
            f"当前 default_value.shape={default_arr.shape}"
            )

    try:
        return np.broadcast_to(default_arr, sample_shape).astype(
            dtype, copy=False
            )
    except ValueError as exc:
        raise ValueError(
            f"default_value 的 shape 无法匹配单个像素的数据维度。"
            f"需要可广播到 {sample_shape}，但得到 {default_arr.shape}"
            ) from exc
