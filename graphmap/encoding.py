"""
底层类型重解释编解码工具

所有编解码操作均基于 numpy 的 view() 实现零拷贝转换。
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# RGBA Color: uint8×4 ⟷ float32
# ═══════════════════════════════════════════════════════════════════════════════

_PACKED_RGBA_DTYPE = np.dtype([("r", np.uint8), ("g", np.uint8),
                                ("b", np.uint8), ("a", np.uint8)])


def pack_rgba(r: int, g: int, b: int, a: int = 255) -> np.float32:
    """
    将 4 个 uint8 值打包为一个 float32 的零拷贝视图。

    Parameters
    ----------
    r, g, b, a : int
        RGBA 通道值，范围 0～255

    Returns
    -------
    np.float32
        打包后的 float32 零拷贝视图
    """
    arr = np.array([(r, g, b, a)], dtype=_PACKED_RGBA_DTYPE)
    return arr.view(np.float32)[0]


def unpack_rgba(packed: np.float32) -> np.ndarray:
    """
    将 float32 零拷贝解包为 4 个 uint8 值。

    Parameters
    ----------
    packed : np.float32
        打包的 float32 值

    Returns
    -------
    np.ndarray
        shape (4,), dtype=uint8, 顺序 [r, g, b, a]
    """
    return np.array([packed],
                    dtype=np.float32).view(_PACKED_RGBA_DTYPE).view(
                        np.uint8
                        )


def pack_rgba_batch(
        rgb: np.ndarray, alpha: np.ndarray = None
    ) -> np.ndarray:
    """
    批量打包 RGB 数组为 float32 数组。

    Parameters
    ----------
    rgb : np.ndarray
        shape (N, 3) uint8 或 shape (N, 4) uint8
    alpha : np.ndarray, optional
        shape (N,) uint8，仅在 rgb 为 (N, 3) 时使用

    Returns
    -------
    np.ndarray
        shape (N,), dtype=float32
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim == 1:
        rgb = rgb.reshape(1, -1)
    N = rgb.shape[0]
    if rgb.shape[1] == 3:
        a = alpha if alpha is not None else np.full(N, 255, dtype=np.uint8)
        rgba = np.stack([rgb[:, 0], rgb[:, 1], rgb[:, 2], a], axis=1)
    else:
        rgba = rgb
    packed = np.zeros(N, dtype=_PACKED_RGBA_DTYPE)
    packed["r"] = rgba[:, 0]
    packed["g"] = rgba[:, 1]
    packed["b"] = rgba[:, 2]
    packed["a"] = rgba[:, 3]
    return packed.view(np.float32)


def unpack_rgba_batch(packed: np.ndarray) -> np.ndarray:
    """
    批量解包 float32 数组为 RGBA uint8 数组。

    Parameters
    ----------
    packed : np.ndarray
        shape (N,), dtype=float32

    Returns
    -------
    np.ndarray
        shape (N, 4), dtype=uint8
    """
    packed = np.asarray(packed, dtype=np.float32)
    N = packed.shape[0]
    return np.ascontiguousarray(packed).view(_PACKED_RGBA_DTYPE).view(
        np.uint8
        ).reshape(N, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Timestamp: int32 ⟷ float32
# ═══════════════════════════════════════════════════════════════════════════════


def pack_timestamp(ms: int) -> np.float32:
    """将 int32 时间戳(ms)零拷贝打包为 float32。"""
    return np.array(ms, dtype=np.int32).view(np.float32).item()


def unpack_timestamp(packed: np.float32) -> int:
    """将 float32 零拷贝解包为 int32 时间戳(ms)。"""
    return int(np.array(packed, dtype=np.float32).view(np.int32).item())


def pack_timestamp_batch(ms_array: np.ndarray) -> np.ndarray:
    """批量打包 int32 时间戳为 float32。"""
    return np.asarray(ms_array, dtype=np.int32).view(np.float32)


def unpack_timestamp_batch(packed: np.ndarray) -> np.ndarray:
    """批量解包 float32 为 int32 时间戳。"""
    return np.asarray(packed, dtype=np.float32).view(np.int32)


# ═══════════════════════════════════════════════════════════════════════════════
# Index: uint32 ⟷ float32
# ═══════════════════════════════════════════════════════════════════════════════


def pack_index(idx: int) -> np.float32:
    """将 uint32 索引零拷贝打包为 float32。"""
    return np.array(idx, dtype=np.uint32).view(np.float32).item()


def unpack_index(packed: np.float32) -> int:
    """将 float32 零拷贝解包为 uint32 索引。"""
    return int(np.array(packed, dtype=np.float32).view(np.uint32).item())


def pack_index_batch(idx_array: np.ndarray) -> np.ndarray:
    """批量打包 uint32 索引为 float32。"""
    return np.asarray(idx_array, dtype=np.uint32).view(np.float32)


def unpack_index_batch(packed: np.ndarray) -> np.ndarray:
    """批量解包 float32 为 uint32 索引。"""
    return np.asarray(packed, dtype=np.float32).view(np.uint32)


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy encoding: int8×4 / uint8×4 ⟷ float32 (保留旧接口兼容)
# ═══════════════════════════════════════════════════════════════════════════════


def encode_4int8_to_1float32(a: int, b: int, c: int, d: int) -> float:
    """
    将 4 个 int8 值打包为一个 float32（零拷贝）。
    仅用于兼容旧代码，新代码建议使用 pack_* 系列。
    """
    arr = np.array([a, b, c, d], dtype=np.int8)
    return arr.view(np.float32).item()


def decode_1float32_to_4int8(packed: float) -> np.ndarray:
    """
    将 float32 解包为 4 个 int8 值（零拷贝）。
    """
    return np.array([packed], dtype=np.float32).view(np.int8).flatten()


def encode_4uint8_to_1float32(a: int, b: int, c: int, d: int) -> float:
    """
    将 4 个 uint8 值打包为一个 float32（零拷贝）。
    与 pack_rgba 语义相同。
    """
    arr = np.array([a, b, c, d], dtype=np.uint8)
    return arr.view(np.float32).item()


def decode_1float32_to_4uint8(packed: float) -> np.ndarray:
    """
    将 float32 解包为 4 个 uint8 值（零拷贝）。
    与 unpack_rgba 语义相同。
    """
    return np.array([packed], dtype=np.float32).view(np.uint8).flatten()


def encode_int32_to_float32(value: int) -> float:
    """将 int32 零拷贝打包为 float32。"""
    return np.array(value, dtype=np.int32).view(np.float32).item()


def decode_float32_to_int32(packed: float) -> int:
    """将 float32 零拷贝解包为 int32。"""
    return int(np.array(packed, dtype=np.float32).view(np.int32).item())


def encode_uint32_to_float32(value: int) -> float:
    """将 uint32 零拷贝打包为 float32。"""
    return np.array(value, dtype=np.uint32).view(np.float32).item()


def decode_float32_to_uint32(packed: float) -> int:
    """将 float32 零拷贝解包为 uint32。"""
    return int(np.array(packed, dtype=np.float32).view(np.uint32).item())
