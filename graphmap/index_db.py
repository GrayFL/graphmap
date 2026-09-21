"""
IndexDB — point-key indexed side tables for graphmap.

Each ``IndexDB`` stores one named table, for example ``semantic_id`` or
``patch_feature``. Map points are addressed by a uint64 point key derived from
voxel coordinates, while repeated values are stored once behind an internal
uint64 content key.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Union

import numpy as np


UINT64_MAX = np.iinfo(np.uint64).max
INT16_MIN = np.iinfo(np.int16).min
INT16_MAX = np.iinfo(np.int16).max


def point_keys_from_xyz(
        xyz: np.ndarray,
        resolution: float,
        legacy_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
    """
    Build uint64 point keys from world coordinates and map resolution.

    Key layout, from high to low bits:
        x:int16 | y:int16 | z:int16 | legacy_hi16
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim == 1:
        xyz = xyz.reshape(1, 3)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")

    coords = np.rint(xyz / float(resolution)).astype(np.int64)
    if np.any(coords < INT16_MIN) or np.any(coords > INT16_MAX):
        bad = coords[
            np.any((coords < INT16_MIN) | (coords > INT16_MAX), axis=1)
            ][0]
        raise ValueError(
            "voxel coordinate exceeds int16 range "
            f"[{INT16_MIN}, {INT16_MAX}]: {bad.tolist()}"
            )

    coords_u16 = coords.astype(np.int16).astype(np.uint16).astype(np.uint64)

    if legacy_indices is None:
        legacy_hi16 = np.zeros(xyz.shape[0], dtype=np.uint64)
    else:
        legacy = np.asarray(legacy_indices, dtype=np.uint32).reshape(-1)
        if legacy.shape[0] != xyz.shape[0]:
            raise ValueError(
                "legacy_indices length must match xyz rows, got "
                f"{legacy.shape[0]} and {xyz.shape[0]}"
                )
        legacy_hi16 = ((legacy.astype(np.uint64) >> 16) & 0xFFFF)

    return (
        (coords_u16[:, 0] << np.uint64(48))
        | (coords_u16[:, 1] << np.uint64(32))
        | (coords_u16[:, 2] << np.uint64(16))
        | legacy_hi16
        ).astype(np.uint64)


def unpack_point_keys(point_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coords_int16_as_int32, legacy_hi16_uint16)`` for keys."""
    keys = np.asarray(point_keys, dtype=np.uint64).reshape(-1)
    parts = np.column_stack([
        ((keys >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.uint16),
        ((keys >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.uint16),
        ((keys >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16),
        ])
    coords = parts.view(np.int16).reshape(-1, 3).astype(np.int32)
    legacy_hi16 = (keys & np.uint64(0xFFFF)).astype(np.uint16)
    return coords, legacy_hi16


class IndexDB:
    """
    A named point-key side table.

    ``point_to_internal`` maps uint64 map point keys to uint64 content keys.
    ``internal_to_points`` provides the reverse lookup. ``records`` stores each
    unique payload exactly once.
    """

    def __init__(
            self,
            table_name: str = "default",
            *,
            _records: Optional[Dict[int, Any]] = None,
            _point_to_internal: Optional[Dict[int, int]] = None,
            _internal_to_points: Optional[Dict[int, Set[int]]] = None,
            _counter: int = 0,
        ):
        self.table_name = str(table_name)
        self.records: Dict[int, Any] = dict(_records or {})
        self.point_to_internal: Dict[int, int] = dict(
            _point_to_internal or {}
            )
        if _internal_to_points is None:
            self.internal_to_points: Dict[int, Set[int]] = {}
            for point_key, internal_key in self.point_to_internal.items():
                self.internal_to_points.setdefault(
                    int(internal_key), set()
                    ).add(int(point_key))
        else:
            self.internal_to_points = {
                int(key): {int(item) for item in points}
                for key, points in _internal_to_points.items()
                }
        self._counter = int(_counter)

    # ── Write APIs ────────────────────────────────────────────────────

    def put(self, point_key: int, value: Any = None, /, **fields) -> int:
        """
        Link one point key to a value and return the internal content key.

        ``fields`` is accepted for legacy call sites and is stored as a dict.
        """
        if fields:
            if value is not None:
                raise ValueError("pass either value or keyword fields, not both")
            value = fields
        return self.link_value(point_key, value)

    def link_value(self, point_key: int, value: Any) -> int:
        point_key = _as_uint64_int(point_key, "point_key")
        internal_key = self._internal_key_for_value(value)
        self.records.setdefault(internal_key, _copy_value(value))

        old_internal = self.point_to_internal.get(point_key)
        if old_internal == internal_key:
            return internal_key
        if old_internal is not None:
            self._unlink_point(point_key, old_internal)

        self.point_to_internal[point_key] = internal_key
        self.internal_to_points.setdefault(internal_key, set()).add(point_key)
        return internal_key

    def merge_from(
            self,
            other: "IndexDB",
            point_key_map: Mapping[int, int],
        ) -> None:
        """
        Merge a temporary/local DB into this table.

        ``point_key_map`` maps the source DB's local point keys to global map
        point keys. Source points absent from the map are ignored.
        """
        for local_key, internal_key in other.point_to_internal.items():
            if int(local_key) not in point_key_map:
                continue
            value = other.records.get(int(internal_key))
            self.link_value(point_key_map[int(local_key)], value)

    def remove_points(self, point_keys: Union[int, np.ndarray, Iterable[int]]):
        """Remove links for one or more point keys and delete zero-ref records."""
        for point_key in _iter_uint64(point_keys):
            internal_key = self.point_to_internal.pop(point_key, None)
            if internal_key is not None:
                self._unlink_point(point_key, internal_key)

    def clear(self):
        self.records.clear()
        self.point_to_internal.clear()
        self.internal_to_points.clear()
        self._counter = 0

    def gc(self, valid_point_keys: np.ndarray):
        """Drop links whose point keys are not in ``valid_point_keys``."""
        valid_set = {int(key) for key in np.asarray(
            valid_point_keys, dtype=np.uint64
            ).reshape(-1)}
        stale = [key for key in self.point_to_internal if key not in valid_set]
        self.remove_points(stale)

    # ── Read APIs ─────────────────────────────────────────────────────

    def get(self, point_key: int, default: Any = None) -> Any:
        internal_key = self.point_to_internal.get(int(point_key))
        if internal_key is None:
            return default
        return self.records.get(internal_key, default)

    def get_batch(
            self,
            point_keys: np.ndarray,
            default: Any = None,
        ) -> List[Any]:
        return [self.get(int(key), default) for key in point_keys]

    def get_internal(self, internal_key: int, default: Any = None) -> Any:
        return self.records.get(int(internal_key), default)

    def internal_key_for(self, point_key: int) -> Optional[int]:
        return self.point_to_internal.get(int(point_key))

    def points_for(self, internal_key: int) -> Set[int]:
        return set(self.internal_to_points.get(int(internal_key), set()))

    def contains(self, point_key: int) -> bool:
        return int(point_key) in self.point_to_internal

    def query_by_field(self, field_name: str) -> Set[int]:
        """Return point keys whose dict payload contains ``field_name``."""
        return {
            point_key
            for point_key, internal_key in self.point_to_internal.items()
            if (
                isinstance(self.records.get(internal_key), dict)
                and field_name in self.records[internal_key]
                )
            }

    def query_by_value(self, field_name: str, value: Any) -> Dict[int, Any]:
        """Return point-key payloads where ``payload[field_name] == value``."""
        result = {}
        for point_key, internal_key in self.point_to_internal.items():
            record = self.records.get(internal_key)
            if (
                isinstance(record, dict)
                and field_name in record
                and _values_equal(record[field_name], value)
            ):
                result[point_key] = record
        return result

    # ── Stats and iteration ───────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of unique stored payloads."""
        return len(self.records)

    @property
    def num_links(self) -> int:
        return len(self.point_to_internal)

    @property
    def indices(self) -> np.ndarray:
        """Sorted point keys."""
        return np.array(sorted(self.point_to_internal), dtype=np.uint64)

    @property
    def internal_indices(self) -> np.ndarray:
        return np.array(sorted(self.records), dtype=np.uint64)

    def next_index(self) -> int:
        """Legacy helper for callers that still need a temporary integer key."""
        idx = self._counter
        while idx in self.point_to_internal or idx in self.records:
            idx += 1
        self._counter = idx + 1
        return idx

    def items(self):
        for point_key in self.keys():
            yield point_key, self.get(point_key)

    def keys(self):
        return self.point_to_internal.keys()

    def values(self):
        for _, value in self.items():
            yield value

    def record_items(self):
        return self.records.items()

    def __len__(self) -> int:
        return self.size

    def __contains__(self, point_key: int) -> bool:
        return self.contains(point_key)

    def __getitem__(self, point_key: int) -> Any:
        value = self.get(point_key, default=None)
        if value is None and int(point_key) not in self.point_to_internal:
            raise KeyError(f"Point key {point_key} not found")
        return value

    def __delitem__(self, point_key: int):
        self.remove_points(point_key)

    # ── Serialization ─────────────────────────────────────────────────

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": 2,
            "table_name": self.table_name,
            "records": {
                str(key): _serialize_value(value)
                for key, value in self.records.items()
                },
            "point_to_internal": {
                str(point): str(internal)
                for point, internal in self.point_to_internal.items()
                },
            "internal_to_points": {
                str(internal): [str(point) for point in sorted(points)]
                for internal, points in self.internal_to_points.items()
                },
            "counter": self._counter,
            }
        with open(path, "wb") as stream:
            pickle.dump(state, stream)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IndexDB":
        path = Path(path)
        with open(path, "rb") as stream:
            state = pickle.load(stream)

        if state.get("version") == 2:
            records = {
                int(key): _deserialize_value(value)
                for key, value in state["records"].items()
                }
            point_to_internal = {
                int(point): int(internal)
                for point, internal in state["point_to_internal"].items()
                }
            internal_to_points = {
                int(internal): {int(point) for point in points}
                for internal, points in state["internal_to_points"].items()
                }
            return cls(
                state.get("table_name", path.stem),
                _records=records,
                _point_to_internal=point_to_internal,
                _internal_to_points=internal_to_points,
                _counter=int(state.get("counter", 0)),
                )

        # Legacy format: records were keyed directly by uint32 point IDs.
        records = {
            int(idx): _deserialize_value(record)
            for idx, record in state["records"].items()
            }
        point_to_internal = {idx: idx for idx in records}
        internal_to_points = {idx: {idx} for idx in records}
        return cls(
            path.stem,
            _records=records,
            _point_to_internal=point_to_internal,
            _internal_to_points=internal_to_points,
            _counter=int(state.get("counter", 0)),
            )

    def __repr__(self) -> str:
        return (
            f"IndexDB(table_name={self.table_name!r}, "
            f"records={self.size}, links={self.num_links})"
            )

    # ── Internal helpers ──────────────────────────────────────────────

    def _internal_key_for_value(self, value: Any) -> int:
        salt = 0
        while True:
            internal_key = _hash_value(value, salt=salt)
            existing = self.records.get(internal_key)
            if existing is None or _values_equal(existing, value):
                return internal_key
            salt += 1

    def _unlink_point(self, point_key: int, internal_key: int):
        refs = self.internal_to_points.get(internal_key)
        if refs is None:
            return
        refs.discard(point_key)
        if not refs:
            self.internal_to_points.pop(internal_key, None)
            self.records.pop(internal_key, None)


def _as_uint64_int(value: int, name: str) -> int:
    result = int(value)
    if result < 0 or result > UINT64_MAX:
        raise ValueError(f"{name} must fit uint64, got {value}")
    return result


def _iter_uint64(values: Union[int, np.ndarray, Iterable[int]]) -> Iterable[int]:
    if isinstance(values, (int, np.integer)):
        yield _as_uint64_int(values, "point_key")
        return
    for value in np.asarray(list(values) if not isinstance(values, np.ndarray)
                            else values, dtype=np.uint64).reshape(-1):
        yield int(value)


def _hash_value(value: Any, *, salt: int = 0) -> int:
    hasher = hashlib.blake2b(digest_size=8, person=b"gm-index")
    hasher.update(salt.to_bytes(8, "little", signed=False))
    hasher.update(_stable_bytes(value))
    return int.from_bytes(hasher.digest(), "big", signed=False)


def _stable_bytes(value: Any) -> bytes:
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        header = (
            f"ndarray:{arr.dtype.str}:{arr.shape}:".encode("utf-8")
            )
        return header + arr.tobytes()
    if isinstance(value, np.generic):
        arr = np.asarray(value)
        return f"scalar:{arr.dtype.str}:".encode("utf-8") + arr.tobytes()
    if isinstance(value, dict):
        parts = [b"dict{"]
        for key in sorted(value.keys(), key=lambda item: repr(item)):
            parts.extend([
                _stable_bytes(key),
                b":",
                _stable_bytes(value[key]),
                b";",
                ])
        parts.append(b"}")
        return b"".join(parts)
    if isinstance(value, tuple):
        return b"tuple(" + b";".join(_stable_bytes(v) for v in value) + b")"
    if isinstance(value, list):
        return b"list[" + b";".join(_stable_bytes(v) for v in value) + b"]"
    if isinstance(value, set):
        items = sorted((_stable_bytes(v) for v in value))
        return b"set{" + b";".join(items) + b"}"
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, (bool, int, float)) or value is None:
        return f"json:{value!r}".encode("utf-8")
    return b"pickle:" + pickle.dumps(value, protocol=5)


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    return _deserialize_value(_serialize_value(value))


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return False
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and np.array_equal(a, b)
            )
    if isinstance(a, dict) or isinstance(b, dict):
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    return a == b


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            "__type__": "ndarray",
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "data": arr.tobytes(),
            }
    if isinstance(value, np.generic):
        return {
            "__type__": "np_scalar",
            "dtype": str(value.dtype),
            "data": np.asarray(value).tobytes(),
            }
    if isinstance(value, dict):
        return {
            "__type__": "dict",
            "items": [
                (_serialize_value(key), _serialize_value(item))
                for key, item in value.items()
                ],
            }
    if isinstance(value, tuple):
        return {
            "__type__": "tuple",
            "items": [_serialize_value(item) for item in value],
            }
    if isinstance(value, list):
        return {
            "__type__": "list",
            "items": [_serialize_value(item) for item in value],
            }
    if isinstance(value, set):
        return {
            "__type__": "set",
            "items": [_serialize_value(item) for item in value],
            }
    return value


def _deserialize_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "__type__" not in value:
        return {
            key: _deserialize_value(item)
            for key, item in value.items()
            }
    value_type = value["__type__"]
    if value_type == "ndarray":
        return np.frombuffer(
            value["data"], dtype=np.dtype(value["dtype"])
            ).copy().reshape(value["shape"])
    if value_type == "np_scalar":
        return np.frombuffer(
            value["data"], dtype=np.dtype(value["dtype"])
            ).copy()[0]
    if value_type == "dict":
        return {
            _deserialize_value(key): _deserialize_value(item)
            for key, item in value["items"]
            }
    if value_type == "tuple":
        return tuple(_deserialize_value(item) for item in value["items"])
    if value_type == "list":
        return [_deserialize_value(item) for item in value["items"]]
    if value_type == "set":
        return {_deserialize_value(item) for item in value["items"]}
    return value
