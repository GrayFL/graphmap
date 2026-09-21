"""Vectorized pose and similarity-transform utilities.

The module uses the active-transform convention

``target = translation + scale * rotation(source)``.

A :class:`Pose` with ``scale == 1`` is an SE(3) rigid transform.  A pose with
positive ``scale != 1`` is an orientation-preserving Sim(3) transform.  The
aliases :data:`Transform`, :data:`Sim3`, and :data:`Similarity3` are provided
for code whose intent is clearer with a transform-oriented name.

Notes
-----
The stored quaternion convention follows SciPy: ``[x, y, z, w]``.  Euler
angles default to the ``xyz`` order and degrees.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal, Sequence, TypeAlias, Union, overload

import numpy as np
from scipy.spatial.transform import Rotation as R

ArrayLike: TypeAlias = Union[np.ndarray, Sequence[float]]
RotationInput: TypeAlias = Union[R, ArrayLike]
RotationType: TypeAlias = Literal["auto",
                                    "euler",
                                    "rotvec",
                                    "quat",
                                    "matrix"]
TransformMode: TypeAlias = Literal[
    "world",
    "position",
    "orientation_world",
    "orientation_local",
    "compose_left",
    "compose_right",
    ]

_TRANSFORM_MODE_ALIASES: dict[str, TransformMode] = {
    "world": "world",
    "frame": "world",
    "similarity": "world",
    "sim3": "world",
    "position": "position",
    "orbit": "position",
    "translation": "position",
    "orientation": "orientation_world",
    "orientation_world": "orientation_world",
    "rotation_world": "orientation_world",
    "orientation_local": "orientation_local",
    "rotation_local": "orientation_local",
    "compose_left": "compose_left",
    "left": "compose_left",
    "group_left": "compose_left",
    "compose_right": "compose_right",
    "right": "compose_right",
    "group_right": "compose_right",
    }


def parse_pose_label(label: str) -> tuple[str, str]:
    """Parse a human-readable pose label.

    Parameters
    ----------
    label : str
        Label in one of the forms ``"camera in world"``, ``"camera / world"``,
        ``"world _ camera"``, ``"world -> camera"``, or ``"world-camera"``.

    Returns
    -------
    wrd_frame : str
        Target, world, or parent frame.
    ego_frame : str
        Source, ego, or child frame.

    Raises
    ------
    TypeError
        If ``label`` is not a string.
    ValueError
        If the label does not match a supported form.
    """
    if not isinstance(label, str):
        raise TypeError("label 必须是字符串")

    label = label.strip()
    # ==========================================
    # 模式 A: Ego 在前 ( <ego> in <wrd> 或 <ego> / <wrd> )
    # ==========================================
    # 解析:
    # (?P<ego>[\w]+) 匹配主体（包含字母、数字、下划线）
    # (?:\s+in\s+|\s*/\s*) 匹配分隔符 " in " 或 "/" (斜杠两边空格可选)
    # (?P<wrd>[\w]+) 匹配基座
    pattern_a = r"^(?P<ego>[\w]+)(?:\s+in\s+|\s*/\s*)(?P<wrd>[\w]+)$"
    # ==========================================
    # 模式 B: Wrd 在前 ( <wrd> _ <ego> 或 <wrd> -> <ego> 或 <wrd> - <ego> )
    # ==========================================
    # 解析:
    # (?:\s+_\s+|\s*(?:->|-)\s*) 匹配分隔符 " _ " 或 "->" 或 "-"
    # (注意 "_" 前后强制需要空格，防止与 base_link 这样的命名冲突)
    pattern_b = r"^(?P<wrd>[\w]+)(?:\s+_\s+|\s*(?:->|-)\s*)(?P<ego>[\w]+)$"

    for pattern in (pattern_a, pattern_b):
        match = re.match(pattern, label)
        if match:
            return match.group("wrd"), match.group("ego")
    raise ValueError(f"无法解析的位姿标签格式: {label!r}")


def _rotation_shape(rotation: R) -> tuple[int, ...]:
    """Return the batch shape of a SciPy rotation object."""
    return rotation.as_quat().shape[:-1]


def _as_vector3(value: ArrayLike, *, name: str) -> np.ndarray:
    """Convert an input to a floating array whose last dimension is three."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} 的最后一维必须为 3，实际形状为 {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须全部为有限数")
    return array


def _normalize_rotation_type(rotation_type: str) -> RotationType:
    """Validate and normalize the rotation representation name."""
    value = rotation_type.lower().strip()
    allowed = {"auto", "euler", "rotvec", "quat", "matrix"}
    if value not in allowed:
        raise ValueError(f"rotation_type 必须是 {sorted(allowed)} 之一")
    return value  # type: ignore[return-value]


def _parse_rotation(
    value: RotationInput | None,
    *,
    expected_batch_shape: tuple[int, ...],
    rotation_type: RotationType = "auto",
    degrees: bool = True,
    euler_order: str = "xyz",
    allow_batch: bool,
    ) -> R:
    """Parse a rotation while resolving the ``(3, 3)`` ambiguity.

    Parameters
    ----------
    value : scipy.spatial.transform.Rotation or array_like or None
        Rotation input.  In automatic mode, the accepted trailing shapes are
        ``(..., 4)`` for quaternions, ``(..., 3, 3)`` for matrices, and
        ``(..., 3)`` for Euler angles.
    expected_batch_shape : tuple of int
        Batch shape inferred from translation.  For :class:`PoseArray`, an
        input whose complete shape is ``expected_batch_shape + (3,)`` is
        interpreted as a batch of Euler angles before checking for a matrix.
        This resolves the special ``(3, 3)`` case.
    rotation_type : {"auto", "euler", "rotvec", "quat", "matrix"}
        Explicit representation or automatic inference.
    degrees : bool
        Whether Euler angles are expressed in degrees.
    euler_order : str
        SciPy Euler axis sequence.
    allow_batch : bool
        Whether a batched rotation is accepted.

    Returns
    -------
    scipy.spatial.transform.Rotation
        Parsed rotation object.

    Raises
    ------
    ValueError
        If the shape, representation, or quaternion norm is invalid.
    """
    rotation_type = _normalize_rotation_type(rotation_type)

    if value is None:
        rotation = R.identity()
    elif isinstance(value, R):
        rotation = value
    else:
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("rotation 必须全部为有限数")

        kind: RotationType = rotation_type
        if kind == "auto":
            if array.ndim >= 1 and array.shape[-1] == 4:
                kind = "quat"
            elif array.shape == expected_batch_shape + (3, ):
                # Important for PoseArray: (3, 3) can mean three Euler triples.
                kind = "euler"
            elif array.ndim >= 2 and array.shape[-2:] == (3, 3):
                kind = "matrix"
            elif array.ndim >= 1 and array.shape[-1] == 3:
                kind = "euler"
            else:
                raise ValueError(
                    "无法自动识别 rotation；支持 (...,4) 四元数、"
                    "(...,3,3) 矩阵和 (...,3) 欧拉角"
                    )

        if kind == "quat":
            if array.ndim == 0 or array.shape[-1] != 4:
                raise ValueError("四元数 rotation 的最后一维必须为 4")
            norms = np.linalg.norm(array, axis=-1)
            if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
                raise ValueError("四元数必须有限且范数非零")
            rotation = R.from_quat(array / norms[..., None])
        elif kind == "matrix":
            if array.ndim < 2 or array.shape[-2:] != (3, 3):
                raise ValueError("旋转矩阵的末两维必须为 (3, 3)")
            rotation = R.from_matrix(array)
        elif kind == "euler":
            if array.ndim == 0 or array.shape[-1] != 3:
                raise ValueError("欧拉角 rotation 的最后一维必须为 3")
            rotation = R.from_euler(euler_order, array, degrees=degrees)
        elif kind == "rotvec":
            if array.ndim == 0 or array.shape[-1] != 3:
                raise ValueError("旋转向量 rotation 的最后一维必须为 3")
            rotation = R.from_rotvec(array)
        else:  # pragma: no cover - guarded by _normalize_rotation_type
            raise RuntimeError(f"未处理的 rotation_type: {kind}")

    if not allow_batch and _rotation_shape(rotation):
        raise ValueError(
            f"Pose 只能接收单个旋转，收到批量形状 {_rotation_shape(rotation)}；"
            "请使用 PoseArray"
            )
    return rotation


def _validate_scale(value: ArrayLike | float) -> np.ndarray:
    """Convert and validate a positive Sim(3) scale array."""
    scale = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("scale 必须有限且严格大于 0")
    return scale


def _broadcast_rotation(rotation: R, shape: tuple[int, ...]) -> R:
    """Broadcast a SciPy rotation to a requested batch shape."""
    quat = np.asarray(rotation.as_quat(), dtype=float)
    current_shape = quat.shape[:-1]
    if len(current_shape) > len(shape):
        raise ValueError(f"旋转批量维度 {current_shape} 无法广播到更低维形状 {shape}")
    reshaped = quat.reshape((1, ) * (len(shape) - len(current_shape))
                            + current_shape + (4, ))
    return R.from_quat(
        np.array(np.broadcast_to(reshaped, shape + (4, )), copy=True)
        )


def _rotate(rotation: R, vectors: np.ndarray) -> np.ndarray:
    """Rotate vectors with NumPy broadcasting over all leading dimensions."""
    matrix = rotation.as_matrix()
    return np.einsum("...ij,...j->...i", matrix, vectors)


def _broadcast_components(
    translation: np.ndarray,
    rotation: R,
    scale: np.ndarray,
    ) -> tuple[np.ndarray, R, np.ndarray, tuple[int, ...]]:
    """Broadcast translation, rotation, and scale to one common batch shape."""
    shape = np.broadcast_shapes(
        translation.shape[:-1], _rotation_shape(rotation), scale.shape
        )
    t = np.array(
        np.broadcast_to(translation, shape + (3, )),
        dtype=float,
        copy=True
        )
    r = _broadcast_rotation(rotation, shape)
    s = np.array(np.broadcast_to(scale, shape), dtype=float, copy=True)
    return t, r, s, shape


def _effective_about_transform(
    transform: "PoseLike", about: ArrayLike | None
    ) -> tuple[np.ndarray, R, np.ndarray, tuple[int, ...]]:
    """Return transform components adjusted to act around a chosen point."""
    if about is None:
        return (
            np.asarray(transform.translation, dtype=float),
            transform.rotation,
            np.asarray(transform.scale, dtype=float),
            transform.shape,
            )

    center = _as_vector3(about, name="about")
    shape = np.broadcast_shapes(transform.shape, center.shape[:-1])
    rotation = _broadcast_rotation(transform.rotation, shape)
    scale = np.broadcast_to(
        np.asarray(transform.scale, dtype=float), shape
        )
    translation = np.broadcast_to(
        np.asarray(transform.translation, dtype=float), shape + (3, )
        )
    center = np.broadcast_to(center, shape + (3, ))
    effective_translation = (
        center + translation
        - scale[..., None] * _rotate(rotation, center)
        )
    return effective_translation, rotation, scale, shape


def _apply_components(
    translation: np.ndarray,
    rotation: R,
    scale: np.ndarray,
    points: np.ndarray,
    *,
    inverse: bool,
    ) -> np.ndarray:
    """Apply broadcast transform components to points."""
    transform_shape = translation.shape[:-1]
    point_shape = points.shape[:-1]
    shape = np.broadcast_shapes(transform_shape, point_shape)

    t = np.broadcast_to(translation, shape + (3, ))
    s = np.broadcast_to(scale, shape)
    r = _broadcast_rotation(rotation, shape)
    p = np.broadcast_to(points, shape + (3, ))

    if inverse:
        return _rotate(r.inv(), (p-t) / s[..., None])
    return t + s[..., None] * _rotate(r, p)


def _compose_components(
    left: "PoseLike", right: "PoseLike"
    ) -> tuple[np.ndarray, R, np.ndarray, tuple[int, ...]]:
    """Compose two SE(3)/Sim(3) objects using exact group multiplication."""
    shape = np.broadcast_shapes(left.shape, right.shape)
    left_t = np.broadcast_to(np.asarray(left.translation), shape + (3, ))
    right_t = np.broadcast_to(
        np.asarray(right.translation), shape + (3, )
        )
    left_s = np.broadcast_to(np.asarray(left.scale), shape)
    right_s = np.broadcast_to(np.asarray(right.scale), shape)
    left_r = _broadcast_rotation(left.rotation, shape)
    right_r = _broadcast_rotation(right.rotation, shape)

    translation = left_t + left_s[..., None] * _rotate(left_r, right_t)
    rotation = left_r * right_r
    scale = left_s * right_s
    return translation, rotation, scale, shape


def _check_composition_frames(left: "PoseLike", right: "PoseLike") -> None:
    """Validate frame labels for exact left-to-right group composition."""
    if (left.ego_frame is not None and right.wrd_frame is not None
            and left.ego_frame != right.wrd_frame):
        raise ValueError(
            "变换坐标系不匹配: "
            f"左侧源坐标系={left.ego_frame!r}, "
            f"右侧目标坐标系={right.wrd_frame!r}"
            )


def _world_result_frames(
        transform: "PoseLike", pose: "PoseLike"
    ) -> tuple[str | None, str | None]:
    """Resolve frame labels for a transform acting in the pose's parent frame."""
    if (transform.ego_frame is not None and pose.wrd_frame is not None
            and transform.ego_frame != pose.wrd_frame):
        raise ValueError(
            "世界系变换坐标系不匹配: "
            f"transform.ego_frame={transform.ego_frame!r}, "
            f"pose.wrd_frame={pose.wrd_frame!r}"
            )
    wrd_frame = transform.wrd_frame or pose.wrd_frame
    return wrd_frame, pose.ego_frame


def _make_pose_like(
    translation: np.ndarray,
    rotation: R,
    scale: np.ndarray,
    *,
    wrd_frame: str | None,
    ego_frame: str | None,
    ) -> "PoseLike":
    """Construct a single pose or pose array according to the batch shape."""
    shape = translation.shape[:-1]
    if not shape:
        return Pose._from_components(
            translation,
            rotation,
            scale,
            wrd_frame=wrd_frame,
            ego_frame=ego_frame,
            )
    return PoseArray._from_components(
        translation,
        rotation,
        scale,
        wrd_frame=wrd_frame,
        ego_frame=ego_frame,
        )


def _normalize_transform_mode(mode: str) -> TransformMode:
    """Normalize a transform mode and its documented aliases."""
    key = mode.lower().strip()
    try:
        return _TRANSFORM_MODE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"未知 transform mode {mode!r}；支持 {sorted(set(_TRANSFORM_MODE_ALIASES.values()))}"
            ) from exc


def _transform_pose(
    pose: "PoseLike",
    transform: "PoseLike",
    *,
    mode: str,
    about: ArrayLike | None,
    ) -> "PoseLike":
    """Implement the shared pose-on-pose transform modes."""
    if not isinstance(transform, (Pose, PoseArray)):
        raise TypeError("transform 必须是 Pose、PoseArray 或它们的别名 Transform")

    normalized_mode = _normalize_transform_mode(mode)

    if normalized_mode == "compose_left":
        adjusted = transform.with_about(
            about
            ) if about is not None else transform
        return adjusted * pose
    if normalized_mode == "compose_right":
        adjusted = transform.with_about(
            about
            ) if about is not None else transform
        return pose * adjusted

    effective_t, effective_r, effective_s, transform_shape = (
        _effective_about_transform(transform, about)
    )
    shape = np.broadcast_shapes(pose.shape, transform_shape)
    pose_t = np.broadcast_to(np.asarray(pose.translation), shape + (3, ))
    pose_s = np.broadcast_to(np.asarray(pose.scale), shape)
    pose_r = _broadcast_rotation(pose.rotation, shape)
    transform_t = np.broadcast_to(effective_t, shape + (3, ))
    transform_s = np.broadcast_to(effective_s, shape)
    transform_r = _broadcast_rotation(effective_r, shape)

    if normalized_mode == "position":
        translation = transform_t + transform_s[
            ..., None] * _rotate(transform_r, pose_t)
        rotation = pose_r
        scale = pose_s
        # Partial active edits stay in the same declared parent frame.
        wrd_frame, ego_frame = pose.wrd_frame, pose.ego_frame
    elif normalized_mode == "world":
        translation = transform_t + transform_s[
            ..., None] * _rotate(transform_r, pose_t)
        rotation = transform_r * pose_r
        # Trajectory/world mapping preserves each pose's local metric scale.
        scale = pose_s
        wrd_frame, ego_frame = _world_result_frames(transform, pose)
    elif normalized_mode == "orientation_world":
        translation = pose_t.copy()
        rotation = transform_r * pose_r
        scale = pose_s
        # Only the represented object's orientation changes; frame labels do not.
        wrd_frame, ego_frame = pose.wrd_frame, pose.ego_frame
    elif normalized_mode == "orientation_local":
        translation = pose_t.copy()
        rotation = pose_r * transform_r
        scale = pose_s
        wrd_frame, ego_frame = pose.wrd_frame, pose.ego_frame
    else:  # pragma: no cover - all enum values handled above
        raise RuntimeError(f"未处理的 transform mode: {normalized_mode}")

    return _make_pose_like(
        translation,
        rotation,
        scale,
        wrd_frame=wrd_frame,
        ego_frame=ego_frame,
        )


class Pose:
    """Represent one SE(3) pose or orientation-preserving Sim(3) transform.

    Parameters
    ----------
    position : array_like, shape (3,), optional
        Translation vector.  ``position`` and ``translation`` are aliases and
        cannot both be supplied.
    rotation : scipy.spatial.transform.Rotation or array_like, optional
        Rotation representation.  With ``rotation_type="auto"``:

        - shape ``(4,)`` is a quaternion ``[x, y, z, w]``;
        - shape ``(3, 3)`` is a rotation matrix;
        - shape ``(3,)`` is an Euler-angle triple, in degrees by default.

    translation : array_like, shape (3,), optional
        Alias of ``position``.
    scale : float, default=1.0
        Positive uniform scale.  ``1`` represents an ordinary rigid pose;
        another value makes the object a Sim(3) transform.
    x, y, z : float, default=0.0
        Translation components used when neither translation alias is given.
    roll, pitch, yaw : float, default=0.0
        Euler angles used when ``rotation`` is omitted.
    degrees : bool, default=True
        Interpret Euler-angle inputs in degrees.
    euler_order : str, default="xyz"
        SciPy Euler axis sequence.
    rotation_type : {"auto", "euler", "rotvec", "quat", "matrix"}, default="auto"
        Explicitly select the rotation representation.  Use ``"rotvec"`` to
        retain rotation-vector input semantics for a three-element array.
    wrd_frame : str, optional
        Target, world, or parent frame name.
    ego_frame : str, optional
        Source, ego, or child frame name.
    label : str, optional
        Human-readable frame label parsed by :func:`parse_pose_label`.

    Notes
    -----
    The active point action is ``y = translation + scale * R(x)``.  Exact group
    composition is available with ``*``.  The :meth:`transform` method exposes
    additional oriented-pose mapping modes that deliberately differ from exact
    group composition.
    """

    def __init__(
        self,
        position: ArrayLike | None = None,
        rotation: RotationInput | None = None,
        *,
        translation: ArrayLike | None = None,
        scale: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        degrees: bool = True,
        euler_order: str = "xyz",
        rotation_type: RotationType = "auto",
        wrd_frame: str | None = None,
        ego_frame: str | None = None,
        label: str | None = None,
        ) -> None:
        if position is not None and translation is not None:
            raise ValueError("position 和 translation 是别名，不能同时传入")
        if label is not None:
            wrd_frame, ego_frame = parse_pose_label(label)

        value = translation if translation is not None else position
        if value is None:
            value = [x, y, z]
        t = _as_vector3(value, name="translation")
        if t.shape != (3, ):
            raise ValueError(
                f"Pose 只能表示单个位姿，translation 必须为 (3,)，实际为 {t.shape}；"
                "批量数据请使用 PoseArray"
                )

        if rotation is None:
            rotation = [roll, pitch, yaw]
            rotation_type = "euler"
        r = _parse_rotation(
            rotation,
            expected_batch_shape=(),
            rotation_type=rotation_type,
            degrees=degrees,
            euler_order=euler_order,
            allow_batch=False,
            )

        s = _validate_scale(scale)
        if s.shape:
            raise ValueError("Pose 的 scale 必须是标量；批量 scale 请使用 PoseArray")

        self._translation = np.array(t, dtype=float, copy=True)
        self._rotation = r
        self._scale = float(s)
        self.wrd_frame = wrd_frame
        self.ego_frame = ego_frame

    @classmethod
    def _from_components(
        cls,
        translation: ArrayLike,
        rotation: R,
        scale: ArrayLike | float,
        *,
        wrd_frame: str | None,
        ego_frame: str | None,
        ) -> "Pose":
        """Construct a pose from already parsed scalar components."""
        obj = cls.__new__(cls)
        t = np.asarray(translation, dtype=float)
        s = np.asarray(scale, dtype=float)
        if t.shape != (3, ) or _rotation_shape(rotation) or s.shape:
            raise ValueError("_from_components 需要单个位姿分量")
        obj._translation = np.array(t, copy=True)
        obj._rotation = rotation
        obj._scale = float(s)
        obj.wrd_frame = wrd_frame
        obj.ego_frame = ego_frame
        return obj

    @classmethod
    def from_about(
        cls,
        *,
        rotation: RotationInput | None = None,
        scale: float = 1.0,
        about: ArrayLike = (0.0, 0.0, 0.0),
        translation: ArrayLike = (0.0, 0.0, 0.0),
        degrees: bool = True,
        euler_order: str = "xyz",
        rotation_type: RotationType = "auto",
        wrd_frame: str | None = None,
        ego_frame: str | None = None,
        ) -> "Pose":
        """Create a transform that rotates and scales around a point.

        Parameters
        ----------
        rotation : scipy.spatial.transform.Rotation or array_like, optional
            Rotation component.
        scale : float, default=1.0
            Positive uniform scale.
        about : array_like, shape (3,), default=(0, 0, 0)
            Center of rotation and scaling.
        translation : array_like, shape (3,), default=(0, 0, 0)
            Additional translation applied after rotating/scaling about the
            center.
        degrees, euler_order, rotation_type
            Rotation parsing options; see :class:`Pose`.
        wrd_frame, ego_frame : str, optional
            Frame labels of the resulting transform.

        Returns
        -------
        Pose
            Transform satisfying
            ``y = about + translation + scale * R(x - about)``.
        """
        raw = cls(
            translation=translation,
            rotation=rotation,
            scale=scale,
            degrees=degrees,
            euler_order=euler_order,
            rotation_type=rotation_type,
            wrd_frame=wrd_frame,
            ego_frame=ego_frame,
            )
        return raw.with_about(about)

    @classmethod
    def from_matrix(
        cls,
        matrix: ArrayLike,
        *,
        wrd_frame: str | None = None,
        ego_frame: str | None = None,
        atol: float = 1e-8,
        ) -> "Pose":
        """Create a pose from a 4-by-4 SE(3) or Sim(3) matrix.

        Parameters
        ----------
        matrix : array_like, shape (4, 4)
            Homogeneous matrix whose upper-left block is ``scale * R``.
        wrd_frame, ego_frame : str, optional
            Frame labels.
        atol : float, default=1e-8
            Numerical tolerance used to validate the homogeneous row and the
            recovered rotation matrix.

        Returns
        -------
        Pose
            Decomposed transform.

        Raises
        ------
        ValueError
            If the matrix is not an orientation-preserving uniform-scale
            homogeneous transform.
        """
        t, r, s = _decompose_similarity_matrix(matrix, atol=atol)
        if t.shape != (3, ):
            raise ValueError("Pose.from_matrix 只接受单个 (4, 4) 矩阵")
        return cls._from_components(
            t, r, s, wrd_frame=wrd_frame, ego_frame=ego_frame
            )

    @property
    def shape(self) -> tuple[()]:
        """tuple: Empty batch shape for a single pose."""
        return ()

    @property
    def single(self) -> bool:
        """bool: Always ``True`` for :class:`Pose`."""
        return True

    @property
    def translation(self) -> np.ndarray:
        """numpy.ndarray: Translation vector with shape ``(3,)``."""
        return self._translation

    @property
    def position(self) -> np.ndarray:
        """numpy.ndarray: Alias of :attr:`translation`."""
        return self._translation

    @property
    def t(self) -> np.ndarray:
        """numpy.ndarray: Short alias of :attr:`translation`."""
        return self._translation

    @property
    def rotation(self) -> R:
        """scipy.spatial.transform.Rotation: Proper rotation component."""
        return self._rotation

    @property
    def orientation(self) -> R:
        """scipy.spatial.transform.Rotation: Alias of :attr:`rotation`."""
        return self._rotation

    @property
    def scale(self) -> float:
        """float: Positive uniform scale."""
        return self._scale

    @property
    def is_rigid(self) -> bool:
        """bool: Whether the transform belongs to SE(3)."""
        return bool(np.isclose(self._scale, 1.0))

    @property
    def matrix(self) -> np.ndarray:
        """numpy.ndarray: Homogeneous SE(3)/Sim(3) matrix with shape ``(4, 4)``."""
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self._scale * self.rotation.as_matrix()
        matrix[:3, 3] = self._translation
        return matrix

    def as_matrix(self) -> np.ndarray:
        """Return the homogeneous SE(3)/Sim(3) matrix.

        Returns
        -------
        numpy.ndarray, shape (4, 4)
            Matrix with ``scale * R`` in the upper-left block.
        """
        return self.matrix

    @property
    def rot_mat(self) -> np.ndarray:
        """numpy.ndarray: Proper rotation matrix without scale."""
        return self.rotation.as_matrix()

    @property
    def quat(self) -> np.ndarray:
        """numpy.ndarray: Quaternion in SciPy ``[x, y, z, w]`` order."""
        return self.rotation.as_quat()

    @property
    def rotvec(self) -> np.ndarray:
        """numpy.ndarray: Rotation vector in radians."""
        return self.rotation.as_rotvec()

    @property
    def euler(self) -> np.ndarray:
        """numpy.ndarray: ``xyz`` Euler angles in degrees."""
        return self.rotation.as_euler("xyz", degrees=True)

    @property
    def euler_rad(self) -> np.ndarray:
        """numpy.ndarray: ``xyz`` Euler angles in radians."""
        return self.rotation.as_euler("xyz", degrees=False)

    @property
    def I(self) -> "Pose":
        """Pose: Exact SE(3)/Sim(3) inverse."""
        return ~self

    def copy(self) -> "Pose":
        """Return a deep copy, including scale and frame labels.

        Returns
        -------
        Pose
            Independent pose object.
        """
        return Pose._from_components(
            self.translation.copy(),
            R.from_quat(self.quat.copy()),
            self.scale,
            wrd_frame=self.wrd_frame,
            ego_frame=self.ego_frame,
            )

    def with_about(self, about: ArrayLike | None) -> "Pose":
        """Return an equivalent transform acting around a chosen point.

        Parameters
        ----------
        about : array_like, shape (3,), or None
            Rotation/scale center.  ``None`` returns a copy.

        Returns
        -------
        Pose
            Transform whose direct point action is
            ``about + translation + scale * R(x - about)``.
        """
        if about is None:
            return self.copy()
        t, r, s, shape = _effective_about_transform(self, about)
        if shape:
            raise ValueError("Pose.with_about 的 about 必须是单个三维点")
        return Pose._from_components(
            t, r, s, wrd_frame=self.wrd_frame, ego_frame=self.ego_frame
            )

    def apply(
            self,
            points: ArrayLike,
            *,
            inverse: bool = False
        ) -> np.ndarray:
        """Apply the transform to one point or an arbitrary point array.

        Parameters
        ----------
        points : array_like, shape (..., 3)
            Input points.
        inverse : bool, default=False
            Apply the inverse transform when ``True``.

        Returns
        -------
        numpy.ndarray, shape (..., 3)
            Transformed points with the input leading shape preserved.
        """
        points_array = _as_vector3(points, name="points")
        return _apply_components(
            self.translation,
            self.rotation,
            np.asarray(self.scale),
            points_array,
            inverse=inverse,
            )

    def ego_to_wrd(self, points: ArrayLike) -> np.ndarray:
        """Transform points from the source/ego frame to the target/world frame.

        Parameters
        ----------
        points : array_like, shape (..., 3)
            Points in the source frame.

        Returns
        -------
        numpy.ndarray, shape (..., 3)
            Points in the target frame.
        """
        return self.apply(points)

    def wrd_to_ego(self, points: ArrayLike) -> np.ndarray:
        """Transform points from the target/world frame to the source/ego frame.

        Parameters
        ----------
        points : array_like, shape (..., 3)
            Points in the target frame.

        Returns
        -------
        numpy.ndarray, shape (..., 3)
            Points in the source frame.
        """
        return self.apply(points, inverse=True)

    def transform(
        self,
        transform: "PoseLike",
        *,
        mode: str = "world",
        about: ArrayLike | None = None,
        ) -> "PoseLike":
        """Apply a general pose/group transform to this pose.

        Parameters
        ----------
        transform : Pose or PoseArray
            Transform operator.  ``Transform`` is an alias of :class:`Pose`,
            so it may contain translation, rotation, and positive scale.
        mode : str, default="world"
            Transformation semantics:

            ``"world"``
                Transform the pose origin by the full transform and left-rotate
                its orientation: ``p' = G(p)``, ``R' = Q R``.  The pose's own
                scale is preserved.  This is the usual trajectory-alignment
                mode and supports a Sim(3) operator without turning every
                rigid pose into a scaled local frame.
            ``"position"``
                Transform only the pose origin.  Orientation and pose scale are
                unchanged.  With ``about`` this performs an orbit around a
                point while preserving absolute orientation.
            ``"orientation_world"``
                Change only orientation using parent/world axes:
                ``R' = Q R``.  Translation and scale of ``transform`` are
                ignored.
            ``"orientation_local"``
                Change only orientation using body/local axes:
                ``R' = R Q``.  Translation and scale of ``transform`` are
                ignored.
            ``"compose_left"``
                Exact group composition ``transform * self``.  Sim(3) scale is
                multiplied into the result.
            ``"compose_right"``
                Exact group composition ``self * transform``.  This is useful
                for local/body-frame offsets and also multiplies Sim(3) scale.

            Aliases include ``"sim3"``/``"frame"`` for ``"world"`` and
            ``"orbit"`` for ``"position"``.
        about : array_like, shape (..., 3), optional
            Center of the operator's rotation and scale.  For
            ``"compose_right"`` the center is interpreted in the pose's local
            frame; for world-oriented modes it is in the parent/world frame.

        Returns
        -------
        Pose or PoseArray
            Transformed pose object.  Broadcasting may turn a single pose into
            a pose array.

        Notes
        -----
        A passive coordinate-frame change can be written by applying the
        inverse operator in ``"world"`` mode.  Exact frame-chain algebra should
        use ``compose_left``, ``compose_right``, or the ``*`` operator.
        """
        return _transform_pose(self, transform, mode=mode, about=about)

    def apply_to_pose(
        self,
        pose: "PoseLike",
        *,
        mode: str = "compose_left",
        about: ArrayLike | None = None,
        ) -> "PoseLike":
        """Use this object as an operator on another pose.

        Parameters
        ----------
        pose : Pose or PoseArray
            Pose to transform.
        mode : str, default="compose_left"
            Mode accepted by :meth:`Pose.transform`.
        about : array_like, shape (..., 3), optional
            Rotation/scale center.

        Returns
        -------
        Pose or PoseArray
            Transformed pose.
        """
        if not isinstance(pose, (Pose, PoseArray)):
            raise TypeError("pose 必须是 Pose 或 PoseArray")
        return pose.transform(self, mode=mode, about=about)

    @overload
    def __call__(self, value: "PoseLike") -> "PoseLike":
        ...

    @overload
    def __call__(self, value: ArrayLike) -> np.ndarray:
        ...

    def __call__(
        self, value: Union["PoseLike", ArrayLike]
        ) -> Union["PoseLike", np.ndarray]:
        """Apply to points, or exactly left-compose when passed a pose."""
        if isinstance(value, (Pose, PoseArray)):
            return self.apply_to_pose(value, mode="compose_left")
        return self.apply(value)

    def __mul__(self, other: "PoseLike") -> "PoseLike":
        """Return exact SE(3)/Sim(3) group composition ``self * other``."""
        if not isinstance(other, (Pose, PoseArray)):
            return NotImplemented
        _check_composition_frames(self, other)
        t, r, s, _ = _compose_components(self, other)
        return _make_pose_like(
            t,
            r,
            s,
            wrd_frame=self.wrd_frame,
            ego_frame=other.ego_frame,
            )

    def __invert__(self) -> "Pose":
        """Return the exact inverse transform."""
        rotation = self.rotation.inv()
        scale = 1.0 / self.scale
        translation = -scale * _rotate(rotation, self.translation)
        return Pose._from_components(
            translation,
            rotation,
            scale,
            wrd_frame=self.ego_frame,
            ego_frame=self.wrd_frame,
            )

    def __truediv__(self, other: "PoseLike") -> "PoseLike":
        """Return ``self * inverse(other)``."""
        if not isinstance(other, (Pose, PoseArray)):
            return NotImplemented
        return self * (~other)

    def allclose(
        self,
        other: "Pose",
        *,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        ) -> bool:
        """Test approximate equality, robust to quaternion sign.

        Parameters
        ----------
        other : Pose
            Pose to compare.
        atol, rtol : float
            Absolute and relative tolerances.

        Returns
        -------
        bool
            ``True`` when translation, scale, and relative rotation agree.
        """
        return bool(
            isinstance(other, Pose) and np.allclose(
                self.translation, other.translation, atol=atol, rtol=rtol
                )
            and np.allclose(self.scale, other.scale, atol=atol, rtol=rtol)
            and np.allclose(
                (self.rotation.inv() * other.rotation).magnitude(),
                0.0,
                atol=atol,
                rtol=rtol,
                )
            )

    def __eq__(self, other: object) -> bool:
        """Return approximate component-wise equality for another pose."""
        return isinstance(other, Pose) and self.allclose(other)

    def __repr__(self) -> str:
        """Return a compact developer-oriented representation."""
        return (
            f"Pose(pos={np.round(self.translation, 3)}, "
            f"euler_deg={np.round(self.euler, 3)}, scale={self.scale:.6g}) "
            f"[{self.wrd_frame} <- {self.ego_frame}]"
            )


class PoseArray:
    """Represent an N-dimensional array of SE(3)/Sim(3) transforms.

    Parameters
    ----------
    position : array_like, shape (..., 3), optional
        Batched translation vectors.  A single ``(3,)`` vector is promoted to
        batch shape ``(1,)``.  ``position`` and ``translation`` are aliases.
    rotation : scipy.spatial.transform.Rotation or array_like, optional
        Batched or broadcastable rotations.  In automatic mode, ``(..., 3)``
        means Euler angles in degrees by default, ``(..., 4)`` means
        quaternions, and ``(..., 3, 3)`` means matrices.
    translation : array_like, shape (..., 3), optional
        Alias of ``position``.
    scale : float or array_like, default=1.0
        Positive uniform scales, broadcast against the translation and rotation
        batch shapes.
    degrees, euler_order, rotation_type
        Rotation parsing options; see :class:`Pose`.
    wrd_frame, ego_frame : str, optional
        Shared frame labels for the complete array.
    label : str, optional
        Human-readable frame label.

    Notes
    -----
    ``PoseArray`` always has at least one batch dimension.  The ambiguous input
    shape ``(3, 3)`` is resolved using translation shape: if translation has
    batch shape ``(3,)``, it is interpreted as three Euler-angle triples;
    otherwise it is interpreted as one rotation matrix and broadcast.  Pass
    ``rotation_type="matrix"`` or ``"euler"`` to override this rule.
    """

    def __init__(
        self,
        position: ArrayLike | None = None,
        rotation: RotationInput | None = None,
        *,
        translation: ArrayLike | None = None,
        scale: ArrayLike | float = 1.0,
        degrees: bool = True,
        euler_order: str = "xyz",
        rotation_type: RotationType = "auto",
        wrd_frame: str | None = None,
        ego_frame: str | None = None,
        label: str | None = None,
        ) -> None:
        if position is not None and translation is not None:
            raise ValueError("position 和 translation 是别名，不能同时传入")
        if label is not None:
            wrd_frame, ego_frame = parse_pose_label(label)

        value = translation if translation is not None else position
        if value is None:
            value = [0.0, 0.0, 0.0]
        t = _as_vector3(value, name="translation")
        if t.shape == (3, ):
            t = t[None, :]
        expected_batch_shape = t.shape[:-1]

        r = _parse_rotation(
            rotation,
            expected_batch_shape=expected_batch_shape,
            rotation_type=rotation_type,
            degrees=degrees,
            euler_order=euler_order,
            allow_batch=True,
            )
        s = _validate_scale(scale)
        t, r, s, shape = _broadcast_components(t, r, s)
        if not shape:
            # Defensive; translation promotion should always prevent this.
            t = t[None, :]
            r = _broadcast_rotation(r, (1, ))
            s = np.asarray([float(s)])
            shape = (1, )

        self._translation = t
        self._rotation = r
        self._scale = s
        self.wrd_frame = wrd_frame
        self.ego_frame = ego_frame

    @classmethod
    def _from_components(
        cls,
        translation: ArrayLike,
        rotation: R,
        scale: ArrayLike | float,
        *,
        wrd_frame: str | None,
        ego_frame: str | None,
        ) -> "PoseArray":
        """Construct a pose array from already parsed components."""
        t = _as_vector3(translation, name="translation")
        s = _validate_scale(scale)
        t, rotation, s, shape = _broadcast_components(t, rotation, s)
        if not shape:
            raise ValueError("PoseArray 至少需要一个批量维度")
        obj = cls.__new__(cls)
        obj._translation = t
        obj._rotation = rotation
        obj._scale = s
        obj.wrd_frame = wrd_frame
        obj.ego_frame = ego_frame
        return obj

    @classmethod
    def from_poses(cls, poses: Iterable[Pose]) -> "PoseArray":
        """Pack single poses into one vectorized array.

        Parameters
        ----------
        poses : iterable of Pose
            Non-empty poses sharing identical frame labels.

        Returns
        -------
        PoseArray
            One-dimensional packed array.

        Raises
        ------
        ValueError
            If the iterable is empty or frame labels differ.
        TypeError
            If an element is not a :class:`Pose`.
        """
        items = list(poses)
        if not items:
            raise ValueError("poses 不能为空")
        if not all(isinstance(pose, Pose) for pose in items):
            raise TypeError("poses 中的所有元素都必须是 Pose")

        wrd_frame = items[0].wrd_frame
        ego_frame = items[0].ego_frame
        if any(pose.wrd_frame != wrd_frame or pose.ego_frame != ego_frame
                for pose in items[1:]):
            raise ValueError("PoseArray 中的位姿必须共享相同的坐标系标签")

        return cls._from_components(
            np.stack([pose.translation for pose in items]),
            R.from_quat(np.stack([pose.quat for pose in items])),
            np.asarray([pose.scale for pose in items]),
            wrd_frame=wrd_frame,
            ego_frame=ego_frame,
            )

    @classmethod
    def concatenate(
            cls,
            arrays: Iterable["PoseArray"],
            axis: int = 0
        ) -> "PoseArray":
        """Concatenate pose arrays along a batch axis.

        Parameters
        ----------
        arrays : iterable of PoseArray
            Non-empty arrays with the same frame labels.
        axis : int, default=0
            Batch axis along which to concatenate.

        Returns
        -------
        PoseArray
            Concatenated result.
        """
        items = list(arrays)
        if not items:
            raise ValueError("arrays 不能为空")
        if not all(isinstance(array, PoseArray) for array in items):
            raise TypeError("arrays 中的所有元素都必须是 PoseArray")

        wrd_frame = items[0].wrd_frame
        ego_frame = items[0].ego_frame
        if any(array.wrd_frame != wrd_frame or array.ego_frame != ego_frame
                for array in items[1:]):
            raise ValueError("待拼接 PoseArray 的坐标系标签必须一致")

        ndim = items[0].ndim
        normalized_axis = axis + ndim if axis < 0 else axis
        if normalized_axis < 0 or normalized_axis >= ndim:
            raise ValueError(
                f"axis={axis} 超出 PoseArray 的批量维度范围 [-{ndim}, {ndim - 1}]"
                )

        return cls._from_components(
            np.concatenate([array.translation for array in items],
                            axis=normalized_axis),
            R.from_quat(
                np.concatenate([array.quat for array in items],
                                axis=normalized_axis)
                ),
            np.concatenate([array.scale for array in items],
                            axis=normalized_axis),
            wrd_frame=wrd_frame,
            ego_frame=ego_frame,
            )

    @classmethod
    def from_about(
        cls,
        *,
        rotation: RotationInput | None = None,
        scale: ArrayLike | float = 1.0,
        about: ArrayLike = (0.0, 0.0, 0.0),
        translation: ArrayLike = (0.0, 0.0, 0.0),
        degrees: bool = True,
        euler_order: str = "xyz",
        rotation_type: RotationType = "auto",
        wrd_frame: str | None = None,
        ego_frame: str | None = None,
        ) -> "PoseArray":
        """Create batched transforms that rotate and scale around points.

        Parameters
        ----------
        rotation, scale, about, translation
            Broadcastable transform components.  Each resulting transform acts
            as ``about + translation + scale * R(x - about)``.
        degrees, euler_order, rotation_type
            Rotation parsing options.
        wrd_frame, ego_frame : str, optional
            Shared frame labels.

        Returns
        -------
        PoseArray
            Batched about-point transforms.
        """
        raw = cls(
            translation=translation,
            rotation=rotation,
            scale=scale,
            degrees=degrees,
            euler_order=euler_order,
            rotation_type=rotation_type,
            wrd_frame=wrd_frame,
            ego_frame=ego_frame,
            )
        result = raw.with_about(about)
        if isinstance(result, Pose):
            return cls.from_poses([result])
        return result

    @classmethod
    def from_matrix(
        cls,
        matrix: ArrayLike,
        *,
        wrd_frame: str | None = None,
        ego_frame: str | None = None,
        atol: float = 1e-8,
        ) -> "PoseArray":
        """Create a pose array from batched homogeneous matrices.

        Parameters
        ----------
        matrix : array_like, shape (..., 4, 4)
            SE(3)/Sim(3) matrices.  A single matrix is promoted to batch one.
        wrd_frame, ego_frame : str, optional
            Shared frame labels.
        atol : float, default=1e-8
            Validation tolerance.

        Returns
        -------
        PoseArray
            Decomposed transforms.
        """
        array = np.asarray(matrix, dtype=float)
        if array.shape == (4, 4):
            array = array[None, ...]
        t, r, s = _decompose_similarity_matrix(array, atol=atol)
        return cls._from_components(
            t, r, s, wrd_frame=wrd_frame, ego_frame=ego_frame
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """tuple of int: Batch shape of the pose array."""
        return self._translation.shape[:-1]

    @property
    def ndim(self) -> int:
        """int: Number of batch dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """int: Total number of poses."""
        return int(np.prod(self.shape))

    @property
    def single(self) -> bool:
        """bool: Always ``False`` for :class:`PoseArray`."""
        return False

    def __len__(self) -> int:
        """Return the size of the first batch axis."""
        return len(self._translation)

    @property
    def translation(self) -> np.ndarray:
        """numpy.ndarray: Batched translations with shape ``shape + (3,)``."""
        return self._translation

    @property
    def position(self) -> np.ndarray:
        """numpy.ndarray: Alias of :attr:`translation`."""
        return self._translation

    @property
    def positions(self) -> np.ndarray:
        """numpy.ndarray: Plural alias of :attr:`translation`."""
        return self._translation

    @property
    def t(self) -> np.ndarray:
        """numpy.ndarray: Short alias of :attr:`translation`."""
        return self._translation

    @property
    def rotation(self) -> R:
        """scipy.spatial.transform.Rotation: Batched proper rotations."""
        return self._rotation

    @property
    def rotations(self) -> R:
        """scipy.spatial.transform.Rotation: Plural alias of :attr:`rotation`."""
        return self._rotation

    @property
    def orientation(self) -> R:
        """scipy.spatial.transform.Rotation: Alias of :attr:`rotation`."""
        return self._rotation

    @property
    def scale(self) -> np.ndarray:
        """numpy.ndarray: Positive uniform scales with shape ``shape``."""
        return self._scale

    @property
    def is_rigid(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating scales numerically equal to one."""
        return np.isclose(self._scale, 1.0)

    @property
    def matrix(self) -> np.ndarray:
        """numpy.ndarray: Batched homogeneous matrices, shape ``shape + (4, 4)``."""
        matrix = np.zeros(self.shape + (4, 4), dtype=float)
        matrix[..., :3, :3] = self.scale[..., None, None] * self.rot_mat
        matrix[..., :3, 3] = self.translation
        matrix[..., 3, 3] = 1.0
        return matrix

    def as_matrix(self) -> np.ndarray:
        """Return batched homogeneous SE(3)/Sim(3) matrices.

        Returns
        -------
        numpy.ndarray, shape (..., 4, 4)
            Homogeneous matrices.
        """
        return self.matrix

    @property
    def rot_mat(self) -> np.ndarray:
        """numpy.ndarray: Batched proper rotation matrices without scale."""
        return self.rotation.as_matrix()

    @property
    def quat(self) -> np.ndarray:
        """numpy.ndarray: Batched quaternions in ``[x, y, z, w]`` order."""
        return self.rotation.as_quat()

    @property
    def rotvec(self) -> np.ndarray:
        """numpy.ndarray: Batched rotation vectors in radians."""
        return self.rotation.as_rotvec()

    @property
    def euler(self) -> np.ndarray:
        """numpy.ndarray: Batched ``xyz`` Euler angles in degrees."""
        return self.rotation.as_euler("xyz", degrees=True)

    @property
    def euler_rad(self) -> np.ndarray:
        """numpy.ndarray: Batched ``xyz`` Euler angles in radians."""
        return self.rotation.as_euler("xyz", degrees=False)

    @property
    def I(self) -> "PoseArray":
        """PoseArray: Exact batched inverse."""
        return ~self

    def copy(self) -> "PoseArray":
        """Return a deep copy, including scales and frame labels.

        Returns
        -------
        PoseArray
            Independent copy.
        """
        return PoseArray._from_components(
            self.translation.copy(),
            R.from_quat(self.quat.copy()),
            self.scale.copy(),
            wrd_frame=self.wrd_frame,
            ego_frame=self.ego_frame,
            )

    def __getitem__(self, index: object) -> "PoseLike":
        """Return a scalar pose or sliced pose array.

        Parameters
        ----------
        index : object
            NumPy-style batch index.

        Returns
        -------
        Pose or PoseArray
            Scalar or batched selection.
        """
        translation = self.translation[index]
        scale = self.scale[index]
        quat = self.quat[index]
        rotation = R.from_quat(quat)
        return _make_pose_like(
            np.asarray(translation),
            rotation,
            np.asarray(scale),
            wrd_frame=self.wrd_frame,
            ego_frame=self.ego_frame,
            )

    def reshape(self, *shape: int | tuple[int, ...]) -> "PoseArray":
        """Return a view-like pose array with a new batch shape.

        Parameters
        ----------
        *shape : int or tuple of int
            New batch shape, following ``numpy.reshape`` conventions.

        Returns
        -------
        PoseArray
            Reshaped pose array.
        """
        if len(shape) == 1 and isinstance(shape[0], tuple):
            target_shape = shape[0]
        else:
            target_shape = tuple(int(value) for value in shape)
        return PoseArray._from_components(
            self.translation.reshape(target_shape + (3, )),
            R.from_quat(self.quat.reshape(target_shape + (4, ))),
            self.scale.reshape(target_shape),
            wrd_frame=self.wrd_frame,
            ego_frame=self.ego_frame,
            )

    def ravel(self) -> "PoseArray":
        """Flatten all batch axes into one dimension.

        Returns
        -------
        PoseArray
            One-dimensional pose array.
        """
        return self.reshape(-1)

    def with_about(self, about: ArrayLike | None) -> "PoseArray":
        """Return equivalent transforms acting around chosen centers.

        Parameters
        ----------
        about : array_like, shape (..., 3), or None
            Broadcastable centers.  ``None`` returns a copy.

        Returns
        -------
        PoseArray
            About-point transforms.
        """
        if about is None:
            return self.copy()
        t, r, s, _ = _effective_about_transform(self, about)
        return PoseArray._from_components(
            t, r, s, wrd_frame=self.wrd_frame, ego_frame=self.ego_frame
            )

    def apply_each(
            self,
            points: ArrayLike,
            *,
            inverse: bool = False
        ) -> np.ndarray:
        """Apply poses to points using ordinary NumPy broadcasting.

        Parameters
        ----------
        points : array_like, shape (..., 3)
            Points whose leading shape is broadcast against :attr:`shape`.
            Typical uses are one point for all poses or one point per pose.
        inverse : bool, default=False
            Apply inverse transforms when ``True``.

        Returns
        -------
        numpy.ndarray, shape broadcast(shape, points.shape[:-1]) + (3,)
            Transformed points.
        """
        points_array = _as_vector3(points, name="points")
        return _apply_components(
            self.translation,
            self.rotation,
            self.scale,
            points_array,
            inverse=inverse,
            )

    def apply_clouds(
            self,
            points: ArrayLike,
            *,
            inverse: bool = False
        ) -> np.ndarray:
        """Apply each pose to its corresponding point cloud without loops.

        Parameters
        ----------
        points : array_like, shape ``pose_batch + cloud_shape + (3,)``
            Per-pose point clouds.  The leading pose-batch axes must be
            broadcast-compatible with :attr:`shape`.
        inverse : bool, default=False
            Apply inverse transforms when ``True``.

        Returns
        -------
        numpy.ndarray
            Array with broadcast pose batch, unchanged cloud axes, and trailing
            coordinate dimension three.

        Raises
        ------
        ValueError
            If the input does not contain enough leading pose-batch axes.
        """
        points_array = _as_vector3(points, name="points")
        point_batch_ndim = points_array.ndim - 1
        if point_batch_ndim < self.ndim:
            raise ValueError(
                "points 的前导维度不足以表示每个 pose 对应的点云；"
                "所有 pose 共享同一点云时请使用 apply_all"
                )
        cloud_ndim = point_batch_ndim - self.ndim
        expanded_shape = self.shape + (1, ) * cloud_ndim
        translation = self.translation.reshape(expanded_shape + (3, ))
        scale = self.scale.reshape(expanded_shape)
        rotation = R.from_quat(self.quat.reshape(expanded_shape + (4, )))
        return _apply_components(
            translation,
            rotation,
            scale,
            points_array,
            inverse=inverse,
            )

    def apply_all(
            self,
            points: ArrayLike,
            *,
            inverse: bool = False
        ) -> np.ndarray:
        """Apply every pose to the same complete point array.

        Parameters
        ----------
        points : array_like, shape ``point_shape + (3,)``
            Shared point or point cloud.
        inverse : bool, default=False
            Apply inverse transforms when ``True``.

        Returns
        -------
        numpy.ndarray, shape ``pose_shape + point_shape + (3,)``
            Outer-product application of all poses to all input points.
        """
        points_array = _as_vector3(points, name="points")
        point_shape = points_array.shape[:-1]
        pose_shape = self.shape
        translation = self.translation.reshape(
            pose_shape + (1, ) * len(point_shape) + (3, )
            )
        scale = self.scale.reshape(pose_shape + (1, ) * len(point_shape))
        rotation = R.from_quat(
            self.quat
            .reshape(pose_shape + (1, ) * len(point_shape) + (4, ))
            )
        points_expanded = points_array.reshape((1, ) * len(pose_shape)
                                                + point_shape + (3, ))
        return _apply_components(
            translation,
            rotation,
            scale,
            points_expanded,
            inverse=inverse,
            )

    def apply(
            self,
            points: ArrayLike,
            *,
            inverse: bool = False
        ) -> np.ndarray:
        """Alias of :meth:`apply_each` for standard broadcasting.

        Parameters
        ----------
        points : array_like, shape (..., 3)
            Broadcastable point array.
        inverse : bool, default=False
            Apply inverse transforms when ``True``.

        Returns
        -------
        numpy.ndarray
            Transformed points.
        """
        return self.apply_each(points, inverse=inverse)

    def ego_to_wrd(self, points: ArrayLike) -> np.ndarray:
        """Apply each forward transform to broadcastable source-frame points."""
        return self.apply_each(points)

    def wrd_to_ego(self, points: ArrayLike) -> np.ndarray:
        """Apply each inverse transform to broadcastable target-frame points."""
        return self.apply_each(points, inverse=True)

    def transform(
        self,
        transform: "PoseLike",
        *,
        mode: str = "world",
        about: ArrayLike | None = None,
        ) -> "PoseLike":
        """Apply general pose/group transform semantics to this pose array.

        Parameters
        ----------
        transform : Pose or PoseArray
            Broadcastable transform operator.
        mode : str, default="world"
            One of ``world``, ``position``, ``orientation_world``,
            ``orientation_local``, ``compose_left``, or ``compose_right``.
            See :meth:`Pose.transform` for complete formulas and Sim(3)
            behavior.
        about : array_like, shape (..., 3), optional
            Broadcastable rotation/scale center.

        Returns
        -------
        PoseArray
            Vectorized transformed poses.  A scalar result is possible only
            through unusual scalar indexing and is returned as :class:`Pose`.
        """
        return _transform_pose(self, transform, mode=mode, about=about)

    def apply_to_pose(
        self,
        pose: "PoseLike",
        *,
        mode: str = "compose_left",
        about: ArrayLike | None = None,
        ) -> "PoseLike":
        """Use this pose array as a broadcast transform operator.

        Parameters
        ----------
        pose : Pose or PoseArray
            Pose target.
        mode : str, default="compose_left"
            Mode accepted by :meth:`Pose.transform`.
        about : array_like, shape (..., 3), optional
            Rotation/scale center.

        Returns
        -------
        Pose or PoseArray
            Transformed result.
        """
        if not isinstance(pose, (Pose, PoseArray)):
            raise TypeError("pose 必须是 Pose 或 PoseArray")
        return pose.transform(self, mode=mode, about=about)

    @overload
    def __call__(self, value: "PoseLike") -> "PoseLike":
        ...

    @overload
    def __call__(self, value: ArrayLike) -> np.ndarray:
        ...

    def __call__(
        self, value: Union["PoseLike", ArrayLike]
        ) -> Union["PoseLike", np.ndarray]:
        """Apply to points, or exactly left-compose when passed poses."""
        if isinstance(value, (Pose, PoseArray)):
            return self.apply_to_pose(value, mode="compose_left")
        return self.apply_each(value)

    def __mul__(self, other: "PoseLike") -> "PoseLike":
        """Return exact broadcast SE(3)/Sim(3) group composition."""
        if not isinstance(other, (Pose, PoseArray)):
            return NotImplemented
        _check_composition_frames(self, other)
        t, r, s, _ = _compose_components(self, other)
        return _make_pose_like(
            t,
            r,
            s,
            wrd_frame=self.wrd_frame,
            ego_frame=other.ego_frame,
            )

    def __invert__(self) -> "PoseArray":
        """Return exact batched inverse transforms."""
        rotation = self.rotation.inv()
        scale = 1.0 / self.scale
        translation = -scale[..., None] * _rotate(
            rotation, self.translation
            )
        return PoseArray._from_components(
            translation,
            rotation,
            scale,
            wrd_frame=self.ego_frame,
            ego_frame=self.wrd_frame,
            )

    def __truediv__(self, other: "PoseLike") -> "PoseLike":
        """Return ``self * inverse(other)`` with broadcasting."""
        if not isinstance(other, (Pose, PoseArray)):
            return NotImplemented
        return self * (~other)

    def allclose(
        self,
        other: "PoseArray",
        *,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        ) -> bool:
        """Test approximate array equality, robust to quaternion sign.

        Parameters
        ----------
        other : PoseArray
            Array to compare.
        atol, rtol : float
            Numerical tolerances.

        Returns
        -------
        bool
            ``True`` when shape and all transform components agree.
        """
        return bool(
            isinstance(other, PoseArray) and self.shape == other.shape
            and np.allclose(
                self.translation, other.translation, atol=atol, rtol=rtol
                )
            and np.allclose(self.scale, other.scale, atol=atol, rtol=rtol)
            and np.allclose(
                (self.rotation.inv() * other.rotation).magnitude(),
                0.0,
                atol=atol,
                rtol=rtol,
                )
            )

    def __eq__(self, other: object) -> bool:
        """Return approximate component-wise equality for another pose array."""
        return isinstance(other, PoseArray) and self.allclose(other)

    def __repr__(self) -> str:
        """Return a compact developer-oriented representation."""
        return (
            f"PoseArray(shape={self.shape}, "
            f"rigid={bool(np.all(self.is_rigid))}, "
            f"frames=[{self.wrd_frame} <- {self.ego_frame}])"
            )


PoseLike: TypeAlias = Union[Pose, PoseArray]


def _decompose_similarity_matrix(
        matrix: ArrayLike, *, atol: float
    ) -> tuple[np.ndarray, R, np.ndarray]:
    """Decompose homogeneous orientation-preserving similarity matrices.

    Parameters
    ----------
    matrix : array_like, shape (..., 4, 4)
        Homogeneous SE(3)/Sim(3) matrix array.
    atol : float
        Validation tolerance.

    Returns
    -------
    translation : numpy.ndarray, shape (..., 3)
        Translation components.
    rotation : scipy.spatial.transform.Rotation
        Proper rotation components.
    scale : numpy.ndarray, shape (...,)
        Positive uniform scales.

    Raises
    ------
    ValueError
        If the matrix has an invalid homogeneous row, reflection/non-positive
        determinant, anisotropic scale, or non-orthonormal recovered rotation.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim < 2 or array.shape[-2:] != (4, 4):
        raise ValueError("matrix 的末两维必须为 (4, 4)")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix 必须全部为有限数")

    expected_row = np.zeros(
        array.shape[:-2] + (4, ), dtype=float
        )
    expected_row[..., 3] = 1.0
    if not np.allclose(array[..., 3, :], expected_row, atol=atol,
                        rtol=0.0):
        raise ValueError("matrix 的最后一行必须为 [0, 0, 0, 1]")

    linear = array[..., :3, :3]
    determinant = np.linalg.det(linear)
    if np.any(determinant <= 0):
        raise ValueError("matrix 必须是方向保持且 scale > 0 的相似变换")
    scale = np.cbrt(determinant)
    rotation_matrix = linear / scale[..., None, None]

    identity = np.eye(3)
    gram = np.swapaxes(rotation_matrix, -1, -2) @ rotation_matrix
    if not np.allclose(gram, identity, atol=atol, rtol=0.0):
        raise ValueError("matrix 的线性部分不是统一尺度乘以正交旋转")
    if not np.allclose(np.linalg.det(rotation_matrix),
                        1.0,
                        atol=atol,
                        rtol=0.0):
        raise ValueError("恢复出的旋转矩阵行列式不为 1")

    return array[..., :3, 3], R.from_matrix(rotation_matrix), scale


# Semantic aliases.  They are intentionally the same class, not subclasses.
Transform = Pose
Sim3 = Pose
Similarity3 = Pose
TransformArray = PoseArray
Sim3Array = PoseArray
Similarity3Array = PoseArray

__all__ = [
    "Pose",
    "PoseArray",
    "PoseLike",
    "Transform",
    "TransformArray",
    "Sim3",
    "Sim3Array",
    "Similarity3",
    "Similarity3Array",
    "parse_pose_label",
    ]
