"""
Environment — 3D point-cloud map manager.

The global map is stored in ``InfoPoints``. Side data is stored in named
``IndexDB`` tables keyed by deterministic uint64 point keys derived from voxel
coordinates.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time as _time
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np

from .index_db import IndexDB, point_keys_from_xyz
from .infopoints import InfoPoints
from .misc.config import Munch, default_config
from .pose import Pose
from .visibility import (
    compute_visible_empty_removal_mask,
    frustum_culling,
    )


class Environment:
    """
    3D semantic-map environment.

    ``InfoPoints.index`` is kept as a legacy/reserved uint32 column. New points
    store zero in that column; live side-table keys are computed from
    coordinates and map resolution.
    """

    def __init__(
        self,
        config: Optional[Munch] = None,
        enable_2d_map: bool = False,
        enable_history: bool = False,
        *,
        resolution: float = 0.05,
        side_length: float = 100.0,
        ):
        self.config = (default_config(resolution=resolution, side_length=side_length)
                       if config is None else Munch.fromDict(config))

        self.infopoints = InfoPoints()
        self.index_dbs: Dict[str, IndexDB] = {}
        self.pose: Optional[Pose] = None
        self._start_time = _time.time()
        self._frame_count = 0

        assert self.config.sensor_params.fov_ref in self.config.sensor_params
        fov_cfg = self.config.sensor_params[
            self.config.sensor_params.fov_ref]
        self.fov_h = fov_cfg.hfov
        self.fov_v = fov_cfg.vfov
        self.min_range = fov_cfg.min_range
        self.max_range = fov_cfg.max_range
        self.near = fov_cfg.near or 0.0
        self.far = fov_cfg.far or 100.0
        self.resolution = float(self.config.map.resolution)
        if not np.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("map resolution must be finite and positive")

        self.enable_2d_map = enable_2d_map
        self._map2d_update_from_points = True
        self.map2d = None
        if enable_2d_map:
            from .map2d import Map2D
            self.map2d = Map2D(config=self.config)

        history_enabled = bool(
            enable_history or self.config.environment.enable_history
            )
        self.history = (
            deque(maxlen=int(self.config.environment.history_length))
            if history_enabled else None
            )

    # ── Compatibility alias ───────────────────────────────────────────

    @property
    def db(self) -> Dict[str, IndexDB]:
        """Migration alias for the named IndexDB table dictionary."""
        return self.index_dbs

    @db.setter
    def db(self, value):
        if isinstance(value, dict):
            self.index_dbs = value
        elif isinstance(value, IndexDB):
            self.index_dbs = {value.table_name: value}
        else:
            raise TypeError("Environment.db must be a dict or IndexDB")

    # ── Core observe API ──────────────────────────────────────────────

    def observe(
        self,
        pose_robot: Pose,
        pose_lidar3d: Pose,
        points_world: np.ndarray,
        points_floor: Optional[np.ndarray],
        points_hwd: np.ndarray = None,
        visible_mask: np.ndarray = None,
        colors: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        extra_keys: Optional[Mapping[str, IndexDB]] = None,
        extra_info: Optional[List[Dict[str, Any]]] = None,
        *,
        update_global: bool = True,
        update_2d: bool = True,
        **kwds,
        ):
        """
        Submit one observation and update the global map.

        ``extra_keys`` maps table names to temporary per-frame ``IndexDB``
        objects. Temporary point keys are local row indices into
        ``points_world``. Only rows that survive global merge/dedup are merged
        into the environment tables.
        """
        del kwds
        self.pose = pose_robot
        self._frame_count += 1

        points_world = np.asarray(points_world, dtype=np.float32)
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError(
                f"points_world must have shape (N, 3), got {points_world.shape}"
                )

        M = points_world.shape[0]
        if M == 0:
            if self.enable_2d_map and update_2d:
                self._update_2d_map(points_world)
            return

        points_floor_arr = (
            np.empty((0, 3), dtype=np.float32) if points_floor is None else
            np.asarray(points_floor, dtype=np.float32).reshape(-1, 3)
            )

        timestamps_ms = self._timestamps_for_observation(timestamp, M)
        colors_arr = self._colors_for_observation(colors, M)

        legacy_zero = np.zeros(M, dtype=np.uint32)
        new_point_keys = point_keys_from_xyz(
            points_world,
            self.resolution,
            legacy_zero,
            )
        new_points = InfoPoints.from_arrays(
            xyz=points_world,
            color=colors_arr,
            timestamp_ms=timestamps_ms,
            indices=legacy_zero,
            )

        if self.history is not None:
            self.history.append({
                "infopoints": new_points,
                "pose_robot": pose_robot,
                "points_floor": points_floor_arr,
                "points_hwd": points_hwd,
                "visible_mask": visible_mask,
                "timestamp": timestamp,
                })

        if not update_global:
            return

        old_point_keys = self.point_keys()
        # keep_old_mask = self._visible_keep_mask(
        #     points_world,
        #     points_floor_arr,
        #     pose_lidar3d,
        #     )
        keep_old_mask = np.ones(self.infopoints.N, dtype=np.bool_)

        old_kept = self.infopoints.filter_by_mask(keep_old_mask)
        old_kept_keys = old_point_keys[keep_old_mask]
        combined = old_kept.concat(new_points)
        combined_keys = np.concatenate([old_kept_keys, new_point_keys])
        combined_sources = np.concatenate([
            np.full(old_kept.N, -1, dtype=np.int64),
            np.arange(M, dtype=np.int64),
            ])

        self.infopoints, final_keys, final_sources = _deduplicate_by_point_key(
            combined,
            combined_keys,
            combined_sources,
            )

        local_to_global = {
            int(local_idx): int(point_key)
            for local_idx, point_key in zip(final_sources, final_keys)
            if int(local_idx) >= 0
            }

        merged_extra = self._normalize_extra_tables(extra_keys, extra_info)
        for table_name, local_db in merged_extra.items():
            self.index_db(table_name).merge_from(local_db, local_to_global)

        for db in self.index_dbs.values():
            db.gc(final_keys)

        if self.enable_2d_map and update_2d:
            self._update_2d_map(points_world)

    # ── Index table helpers ───────────────────────────────────────────

    def index_db(self, name: str) -> IndexDB:
        name = str(name)
        if name not in self.index_dbs:
            self.index_dbs[name] = IndexDB(name)
        return self.index_dbs[name]

    def point_keys(
        self,
        points: Optional[Union[InfoPoints, np.ndarray]] = None,
        ) -> np.ndarray:
        if points is None:
            return point_keys_from_xyz(
                self.infopoints.xyz,
                self.resolution,
                self.infopoints.index,
                )
        if isinstance(points, InfoPoints):
            return point_keys_from_xyz(
                points.xyz,
                self.resolution,
                points.index,
                )

        arr = np.asarray(points)
        if arr.ndim == 1:
            if arr.size == 3 and arr.dtype.kind in "fc":
                return point_keys_from_xyz(
                    arr.reshape(1, 3), self.resolution
                    )
            return arr.astype(np.uint64).reshape(-1)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return point_keys_from_xyz(arr, self.resolution)
        return arr.astype(np.uint64).reshape(-1)

    def get_index_values(
        self,
        name: str,
        points: Optional[Union[InfoPoints, np.ndarray]] = None,
        default: Any = None,
        ) -> List[Any]:
        keys = self.point_keys(points)
        db = self.index_dbs.get(str(name))
        if db is None:
            return [default for _ in range(len(keys))]
        return db.get_batch(keys, default=default)

    def set_point_attributes(self, attributes: Mapping[str, Any]):
        """Replace derived tables aligned with the CURRENT fused cloud rows.

        Unlike observe(extra_keys=...), keys here are already global voxel keys.
        Recompute full-cloud properties such as ground confidence after geometry
        or connectivity changes, including previously observed points. Other
        tables (e.g. semantic IDs) are preserved. Validation precedes mutation.
        """
        keys = self.point_keys()
        replacement = {}
        for name, values in attributes.items():
            if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError("attribute names must be nonempty file-safe table names")
            if len(values) != self.N:
                raise ValueError(f"attributes[{name!r}] must align with all {self.N} map points")
            table = IndexDB(name)
            for key, value in zip(keys, values):
                table.put(int(key), value.item() if isinstance(value, np.generic) else value)
            replacement[name] = table
        self.index_dbs.update(replacement)

    def set_map2d(self, map2d, *, update_from_points=False):
        """Attach a derived grid without legacy occupancy updates overwriting it.

        ``update_from_points=True`` opts into the original raw point projection.
        This policy is saved with the environment and restored on load.
        """
        from .map2d import Map2D
        if map2d is not None and not isinstance(map2d, Map2D):
            raise TypeError("map2d must be Map2D or None")
        self.map2d = map2d
        self.enable_2d_map = map2d is not None
        self._map2d_update_from_points = bool(update_from_points)

    # ── Internal helpers ──────────────────────────────────────────────

    def _timestamps_for_observation(
        self,
        timestamp: Optional[float],
        count: int,
        ) -> np.ndarray:
        if timestamp is None:
            now_ms = int((_time.time() - self._start_time) * 1000)
            return np.full(count, now_ms, dtype=np.int32)
        if isinstance(timestamp, (float, int, np.floating, np.integer)):
            timestamp_ms = int(float(timestamp) * 1000)
            return np.full(count, timestamp_ms, dtype=np.int32)
        timestamps = np.asarray(timestamp, dtype=np.float64).reshape(-1)
        if len(timestamps) != count:
            raise ValueError(
                f"timestamp array length {len(timestamps)} != point count {count}"
                )
        return (timestamps * 1000).astype(np.int32)

    def _colors_for_observation(
        self,
        colors: Optional[np.ndarray],
        count: int,
        ) -> np.ndarray:
        if colors is None:
            return np.full((count, 4), 128, dtype=np.uint8)

        colors_arr = np.asarray(colors, dtype=np.uint8)
        if colors_arr.ndim == 1:
            colors_arr = np.tile(colors_arr.reshape(1, -1), (count, 1))
        if colors_arr.shape[0] != count:
            raise ValueError(
                f"colors length {colors_arr.shape[0]} != point count {count}"
                )
        if colors_arr.shape[1] == 3:
            alpha = np.full((count, 1), 255, dtype=np.uint8)
            return np.hstack([colors_arr, alpha])
        if colors_arr.shape[1] != 4:
            raise ValueError(
                f"colors must have 3 or 4 channels, got {colors_arr.shape[1]}"
                )
        return colors_arr

    def _visible_keep_mask(
        self,
        points_world: np.ndarray,
        points_floor: np.ndarray,
        pose_lidar3d: Pose,
        ) -> np.ndarray:
        if self.infopoints.N == 0:
            return np.zeros(0, dtype=bool)

        pts_new = (
            np.concatenate((points_floor, points_world), axis=0)
            if len(points_floor) else points_world
            )
        remove_old = compute_visible_empty_removal_mask(
            pts_world=self.infopoints.xyz,
            pts_new=pts_new,
            pose_lidar=pose_lidar3d,
            hfov_update=self.fov_h,
            vfov_update=self.fov_v,
            hres=0.36,
            vres=0.36,
            min_range=self.near,
            max_range=self.far,
            voxel_size=self.resolution,
            free_base_margin=0.0,
            free_voxel_factor=0.0,
            )
        return ~remove_old

    def _normalize_extra_tables(
        self,
        extra_keys: Optional[Mapping[str, IndexDB]],
        extra_info: Optional[List[Dict[str, Any]]],
        ) -> Dict[str, IndexDB]:
        tables = dict(extra_keys or {})

        # Legacy compatibility: list-of-dicts point metadata becomes one table.
        if extra_info is not None and "extra_info" not in tables:
            legacy = IndexDB("extra_info")
            for local_idx, info in enumerate(extra_info):
                if info is not None:
                    legacy.put(local_idx, info)
            tables["extra_info"] = legacy

        for table_name, table in tables.items():
            if not isinstance(table, IndexDB):
                raise TypeError(
                    f"extra_keys[{table_name!r}] must be an IndexDB, "
                    f"got {type(table).__name__}"
                    )
        return tables

    def _update_2d_map(self, points_world: np.ndarray):
        if self.map2d is not None and self._map2d_update_from_points:
            self.map2d.update_from_occupancy(points_world, layer=0)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def N(self) -> int:
        return self.infopoints.N

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ── Save/load ─────────────────────────────────────────────────────

    def save(self, dir_path: Union[str, Path]):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.infopoints.save(dir_path / "infopoints.npz")

        index_dir = dir_path / "index_dbs"
        index_dir.mkdir(parents=True, exist_ok=True)
        for table_name, db in self.index_dbs.items():
            db.save(index_dir / f"{table_name}.pkl")

        if self.map2d is not None:
            self.map2d.save(dir_path / "map2d.npz")

        metadata = {
            "map2d_update_from_points": self._map2d_update_from_points,
            "resolution_3d":
                self.resolution,
            "fov_h":
                np.asarray(self.fov_h, dtype=object),
            "fov_v":
                np.asarray(self.fov_v, dtype=object),
            "min_range":
                self.min_range,
            "max_range":
                self.max_range,
            "near":
                self.near,
            "far":
                self.far,
            "frame_count":
                self._frame_count,
            "index_db_tables":
                np.asarray(sorted(self.index_dbs.keys()), dtype=object),
            }
        np.savez(dir_path / "metadata.npz", **metadata)

    @classmethod
    def load(
        cls,
        dir_path: Union[str, Path],
        config: Optional[Munch] = None,
        ) -> "Environment":
        dir_path = Path(dir_path)
        meta = np.load(dir_path / "metadata.npz", allow_pickle=True)
        if config is None:
            config = _config_from_metadata(meta)

        map2d_path = dir_path / "map2d.npz"
        instance = cls(config, enable_2d_map=map2d_path.exists())
        instance.infopoints = InfoPoints.load(dir_path / "infopoints.npz")

        index_dir = dir_path / "index_dbs"
        if index_dir.exists():
            table_names = _metadata_value(meta, "index_db_tables", None)
            if table_names is None:
                paths = sorted(index_dir.glob("*.pkl"))
            else:
                paths = [index_dir / f"{name}.pkl" for name in table_names]
            for path in paths:
                if path.exists():
                    db = IndexDB.load(path)
                    instance.index_dbs[db.table_name] = db
        elif (dir_path / "index_db.pkl").exists():
            instance.index_dbs["legacy"] = instance._migrate_legacy_db(
                IndexDB.load(dir_path / "index_db.pkl")
                )

        if map2d_path.exists():
            from .map2d import Map2D
            instance.map2d = Map2D.load(map2d_path)
            instance.map2d.config = instance.config

        instance._map2d_update_from_points = bool(
            _metadata_value(meta, "map2d_update_from_points", True))

        instance._frame_count = int(
            _metadata_value(meta, "frame_count", 0)
            )
        return instance

    def _migrate_legacy_db(self, legacy_db: IndexDB) -> IndexDB:
        migrated = IndexDB("legacy")
        if self.infopoints.N == 0:
            return migrated
        point_keys = self.point_keys()
        for legacy_index, point_key in zip(self.infopoints.index, point_keys):
            legacy_index_int = int(legacy_index)
            if legacy_index_int in legacy_db:
                migrated.put(int(point_key), legacy_db[legacy_index_int])
        return migrated

    # ── Utility APIs ──────────────────────────────────────────────────

    def get_visible_points(
        self,
        pose: Optional[Pose] = None,
        fov_h: Optional[float] = None,
        fov_v: Optional[float] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        ) -> InfoPoints:
        if pose is None:
            pose = self.pose
        if pose is None or self.infopoints.N == 0:
            return InfoPoints()

        mask = frustum_culling(
            self.infopoints.xyz,
            pose,
            fov_h=self.fov_h if fov_h is None else fov_h,
            fov_v=self.fov_v if fov_v is None else fov_v,
            near=self.near if near is None else near,
            far=self.far if far is None else far,
            )
        return self.infopoints.filter_by_mask(mask)

    def status(self) -> dict:
        return {
            "num_points":
                self.N,
            "num_db_tables":
                len(self.index_dbs),
            "num_db_records":
                sum(db.size for db in self.index_dbs.values()),
            "num_db_links":
                sum(db.num_links for db in self.index_dbs.values()),
            "db_tables": {
                name: {
                    "records": db.size, "links": db.num_links
                    }
                for name, db in sorted(self.index_dbs.items())
                },
            "frame_count":
                self._frame_count,
            "has_2d_map":
                self.map2d is not None,
            "last_pose":
                str(self.pose) if self.pose else None,
            }

    def __repr__(self) -> str:
        return (
            f"Environment(points={self.N}, tables={len(self.index_dbs)}, "
            f"frames={self._frame_count})"
            )


def _deduplicate_by_point_key(
    points: InfoPoints,
    point_keys: np.ndarray,
    source_indices: np.ndarray,
    ) -> tuple[InfoPoints, np.ndarray, np.ndarray]:
    if points.N == 0:
        return InfoPoints(), np.empty(0, dtype=np.uint64), source_indices[:0]

    point_keys = np.asarray(point_keys, dtype=np.uint64).reshape(-1)
    source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if len(point_keys) != points.N or len(source_indices) != points.N:
        raise ValueError(
            "point_keys/source_indices length must match points"
            )

    times = points.timestamp.astype(np.int64)
    positions = np.arange(points.N, dtype=np.int64)
    order = np.lexsort((-positions, -times, point_keys))
    ordered_keys = point_keys[order]
    _, first = np.unique(ordered_keys, return_index=True)
    keep = np.sort(order[first])

    return (
        InfoPoints(points.data[keep], _safe=True),
        point_keys[keep],
        source_indices[keep],
        )


def _metadata_value(meta, name: str, default: Any) -> Any:
    if name not in meta:
        return default
    value = meta[name]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    return value


def _config_from_metadata(meta) -> Munch:
    fov_h = _metadata_value(meta, "fov_h", 90.0)
    fov_v = _metadata_value(meta, "fov_v", 60.0)
    near = float(_metadata_value(meta, "near", 0.0))
    far = float(_metadata_value(meta, "far", 100.0))
    resolution = float(_metadata_value(meta, "resolution_3d", 0.05))
    return Munch.fromDict({
        "sensor_params": {
            "fov_ref": "loaded_sensor",
            "loaded_sensor": {
                "hfov": fov_h,
                "vfov": fov_v,
                "min_range": near,
                "max_range": far,
                "near": near,
                "far": far,
                },
            },
        "map": {
            "resolution": resolution,
            "side_length": 0.0,
            },
        "environment": {
            "enable_history": False,
            "history_length": 1,
            },
        })
