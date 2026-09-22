"""Sampling of aligned H-W images, labels, and feature maps."""
from typing import Any, Optional, Literal
import numpy as np
from ..misc.config import Munch


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
