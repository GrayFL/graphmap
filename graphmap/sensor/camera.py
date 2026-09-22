"""Pinhole intrinsics in graphmap's X-forward camera frame."""
from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Union
import numpy as np


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
            默认按行排列；order="center" 按到光心的距离由近到远排列。

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

            if not all(isinstance(value, Integral) for value in stride):
                raise ValueError("stride 必须为正整数")
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

