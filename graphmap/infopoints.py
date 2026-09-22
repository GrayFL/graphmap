"""
InfoPoints — 3D 点云语义地图核心数据结构

形状 (N, 6)，dtype=np.float32

列布局:
    0: x       (float32)  世界坐标 X
    1: y       (float32)  世界坐标 Y
    2: z       (float32)  世界坐标 Z
    3: color   (float32)  uint8×4 (r, g, b, a) 打包编码
    4: time    (float32)  int32 (ms) 打包编码，从程序启动基准点计时
    5: index   (float32)  uint32 打包编码，legacy/reserved，新增点写 0
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict

from .encoding import (
    pack_rgba_batch,
    unpack_rgba_batch,
    pack_timestamp_batch,
    unpack_timestamp_batch,
    pack_index_batch,
    unpack_index_batch,
    unpack_rgba,
    unpack_timestamp,
    unpack_index,
    )

# 列索引常量
COL_LAYOUT: Dict[str, int] = {
    "x": 0,
    "y": 1,
    "z": 2,
    "color": 3,
    "time": 4,
    "index": 5,
    }
"""InfoPoints 中各字段的列索引"""

NUM_COLS: int = 6
"""InfoPoints 总列数"""


class InfoPoints:
    """
    3D 点云地图的核心存储结构。

    存储格式为 (N, 6) float32 连续数组，无需拆分。
    通过属性访问器按需零拷贝转换为原生类型。
    第 6 列 ``index`` 仅作为旧格式保留；新的 IndexDB 通过点坐标
    和地图分辨率计算 uint64 point key。
    """

    def __init__(
            self,
            data: Optional[np.ndarray] = None,
            *,
            metadata: Optional[Dict[str, str]] = None,
            _safe: bool = False
        ):
        """
        Parameters
        ----------
        data : np.ndarray, optional
            shape (N, 6), dtype=np.float32 的预构建数组
        _safe : bool
            内部标志，跳过检查（用于 from_* 构造器）
        """
        if data is None:
            data = np.empty((0, NUM_COLS), dtype=np.float32)
        elif not _safe:
            data = np.asarray(data, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] != NUM_COLS:
                raise ValueError(
                    f"InfoPoints data must have {NUM_COLS} columns, "
                    f"got {data.shape[1]}"
                    )
        self._data = data
        self._metadata = metadata or {}

    # ── 内部裸数组访问 ────────────────────────────────────────────────

    @property
    def data(self) -> np.ndarray:
        """返回底层 (N, 6) float32 数组（零拷贝，可修改）。"""
        return self._data

    @property
    def N(self) -> int:
        """点的数量。"""
        return self._data.shape[0]

    @property
    def shape(self) -> tuple:
        """返回 (N, 6)。"""
        return self._data.shape

    # ── 字段属性访问器 ────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, str]:
        """返回元数据字典。"""
        return self._metadata

    @property
    def xyz(self) -> np.ndarray:
        """shape (N, 3), dtype=float32 世界坐标。"""
        return self._data[:, :3]

    @xyz.setter
    def xyz(self, value: np.ndarray):
        self._data[:, :3] = np.asarray(value, dtype=np.float32)

    @property
    def color(self) -> np.ndarray:
        """shape (N, 4), dtype=uint8 RGBA。零拷贝视图。"""
        return unpack_rgba_batch(self._data[:, 3])

    @color.setter
    def color(self, value: np.ndarray):
        self._data[:, 3] = pack_rgba_batch(value)

    @property
    def timestamp(self) -> np.ndarray:
        """shape (N,), dtype=int32 时间戳(ms)。零拷贝视图。"""
        return unpack_timestamp_batch(self._data[:, 4])

    @timestamp.setter
    def timestamp(self, value: np.ndarray):
        self._data[:, 4] = pack_timestamp_batch(value)

    @property
    def timestamp_ms(self) -> np.ndarray:
        """timestamp 的别名。"""
        return self.timestamp

    @timestamp_ms.setter
    def timestamp_ms(self, value: np.ndarray):
        self.timestamp = value

    @property
    def index(self) -> np.ndarray:
        """shape (N,), dtype=uint32 legacy/reserved 索引。零拷贝视图。"""
        return unpack_index_batch(self._data[:, 5])

    @index.setter
    def index(self, value: np.ndarray):
        self._data[:, 5] = pack_index_batch(value)

    @property
    def indices(self) -> np.ndarray:
        """index 的别名。"""
        return self.index

    @indices.setter
    def indices(self, value: np.ndarray):
        self.index = value

    # ── 单点访问 ──────────────────────────────────────────────────────

    def get_point(self, idx: int) -> dict:
        """按索引获取单点的所有字段（标量值）。"""
        if idx < 0 or idx >= self.N:
            raise IndexError(f"Index {idx} out of range [0, {self.N})")
        row = self._data[idx]
        color = unpack_rgba(row[3])
        return {
            "x": float(row[0]),
            "y": float(row[1]),
            "z": float(row[2]),
            "r": int(color[0]),
            "g": int(color[1]),
            "b": int(color[2]),
            "a": int(color[3]),
            "timestamp_ms": int(unpack_timestamp(row[4])),
            "index": int(unpack_index(row[5])),
            }

    # ── 切片与索引 ────────────────────────────────────────────────────

    def __getitem__(self, key) -> "InfoPoints":
        """支持整数索引、切片、布尔/整数数组索引。"""
        data = self._data[key]
        if data.ndim == 1:
            data = data.reshape(1, NUM_COLS)
        return InfoPoints(data, _safe=True, metadata=self._metadata.copy())

    def __len__(self) -> int:
        return self.N

    # ── 构造器 ────────────────────────────────────────────────────────

    @classmethod
    def from_arrays(
            cls,
            xyz: np.ndarray,
            color: np.ndarray,
            timestamp_ms: np.ndarray,
            indices: Optional[np.ndarray] = None
        ) -> "InfoPoints":
        """
        从分离的数组构造 InfoPoints。

        Parameters
        ----------
        xyz : np.ndarray
            shape (N, 3) float32
        color : np.ndarray
            shape (N, 3) 或 (N, 4) uint8
        timestamp_ms : np.ndarray
            shape (N,) int32
        indices : np.ndarray, optional
            shape (N,) uint32 legacy/reserved index。未提供时写入 0。
        """
        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, 3)
        N = xyz.shape[0]

        data = np.empty((N, NUM_COLS), dtype=np.float32)
        data[:, :3] = xyz
        data[:, 3] = pack_rgba_batch(np.asarray(color, dtype=np.uint8))
        data[:, 4] = pack_timestamp_batch(
            np.asarray(timestamp_ms, dtype=np.int32)
            )
        if indices is None:
            indices = np.zeros(N, dtype=np.uint32)
        data[:, 5] = pack_index_batch(np.asarray(indices, dtype=np.uint32))
        return cls(data, _safe=True)

    @classmethod
    def empty(cls) -> "InfoPoints":
        """构造空 InfoPoints。"""
        return cls()

    # ── 合并与去重 ────────────────────────────────────────────────────

    def concat(self, other: "InfoPoints") -> "InfoPoints":
        """返回两个 InfoPoints 的拼接；去重需另行调用 deduplicate。"""
        if other.N == 0:
            return InfoPoints(self._data.copy(), _safe=True)
        if self.N == 0:
            return InfoPoints(other._data.copy(), _safe=True)
        return InfoPoints(
            np.concatenate([self._data, other._data], axis=0), _safe=True
            )

    def filter_by_index(self, indices: np.ndarray) -> "InfoPoints":
        """保留 index 在给定集合中的点。"""
        indices = np.asarray(indices, dtype=np.uint32)
        cur = self.index
        mask = np.isin(cur, indices)
        return InfoPoints(self._data[mask], _safe=True)

    def filter_by_mask(self, mask: np.ndarray) -> "InfoPoints":
        """按布尔掩码过滤。"""
        mask = np.asarray(mask, dtype=bool)
        return InfoPoints(self._data[mask], _safe=True)

    def deduplicate(self, resolution: float = 0.05) -> "InfoPoints":
        """
        基于坐标量化的去重（保留最新时间戳的点）。

        Parameters
        ----------
        resolution : float
            体素分辨率，坐标会被量化到此倍数的网格上

        Returns
        -------
        InfoPoints
            去重后的点
        """
        if self.N == 0:
            return InfoPoints()

        xyz = self.xyz
        coords = np.round(xyz / resolution).astype(np.int32)

        # 使用结构化数组找到每个网格最近的点
        dt = np.dtype([("x", np.int32), ("y", np.int32), ("z", np.int32)])
        structured = coords.view(dt).ravel()

        times = self.timestamp

        # 按坐标+时间排序，时间降序使每个网格的第一个点是最新的
        sort_key = np.lexsort(
            (-times, coords[:, 2], coords[:, 1], coords[:, 0])
            )

        sorted_struct = structured[sort_key]
        # 找到每个唯一网格的第一个出现位置
        _, unique_idx = np.unique(sorted_struct, return_index=True)
        keep_idx = sort_key[unique_idx]

        return InfoPoints(self._data[keep_idx], _safe=True)

    # ── 序列化 ────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]):
        """保存到 .npz 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, data=self._data, metadata=self._metadata)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "InfoPoints":
        """从 .npz 文件加载。"""
        path = Path(path)
        # Legacy metadata is an object scalar. Load trusted snapshots only,
        # as with IndexDB's pickle format.
        with np.load(path, allow_pickle=True) as loaded:
            metadata = loaded["metadata"].item() if "metadata" in loaded else {}
            return cls(
                loaded["data"], _safe=True, metadata=metadata
                )

    # ── 实用方法 ──────────────────────────────────────────────────────

    def copy(self) -> "InfoPoints":
        """返回完整拷贝。"""
        return InfoPoints(
            self._data.copy(), _safe=True, metadata=self._metadata.copy()
            )

    def __repr__(self) -> str:
        return f"InfoPoints(N={self.N}, shape={self.shape})"
