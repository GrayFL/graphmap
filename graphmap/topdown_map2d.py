"""Graph-aligned three-layer Habitat top-down map."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .encoding import pack_rgba_batch, unpack_rgba_batch
from .map2d import Map2D

HABITAT_NAVIGATION_LAYER = 0
HABITAT_COLOR_LAYER = 1
ONLINE_OCCUPANCY_LAYER = 2
TOPDOWN_NUM_LAYERS = 3


class TopdownMap2D(Map2D):
    """NavMesh, orthographic color, and online occupancy in one grid."""

    def __init__(
            self,
            *,
            resolution: float,
            shape: tuple[int, int],
            origin: tuple[float, float],
            habitat_bounds: np.ndarray | None = None,
            graph_bounds: np.ndarray | None = None,
            floor_height: float = 0.0
        ):
        super().__init__(
            config=None,
            resolution=resolution,
            side_length=max(shape) * resolution,
            num_layers=TOPDOWN_NUM_LAYERS,
            shape=shape,
            origin=origin,
            )
        self.habitat_bounds = (
            None if habitat_bounds is None else
            np.asarray(habitat_bounds, dtype=np.float32).reshape(2, 3)
            )
        self.graph_bounds = (
            None if graph_bounds is None else
            np.asarray(graph_bounds, dtype=np.float32).reshape(2, 2)
            )
        self.floor_height = float(floor_height)

    @classmethod
    def from_habitat_maps(
            cls,
            navigation: np.ndarray,
            color: np.ndarray,
            habitat_bounds: np.ndarray,
            resolution: float,
            floor_height: float
        ) -> "TopdownMap2D":
        navigation = cls._validate_navigation(navigation)
        color = cls._validate_color(color, navigation.shape)
        habitat_bounds = np.asarray(
            habitat_bounds, dtype=np.float32
            ).reshape(2, 3)
        lower, upper = habitat_bounds
        graph_bounds = np.array(
            [
                [-upper[2], -upper[0]],
                [-lower[2], -lower[0]],
                ],
            dtype=np.float32,
            )
        origin = tuple((-graph_bounds[0] / float(resolution)).tolist())
        result = cls(
            resolution=float(resolution),
            shape=navigation.shape,
            origin=origin,
            habitat_bounds=habitat_bounds,
            graph_bounds=graph_bounds,
            floor_height=floor_height,
            )
        result.set_habitat_navigation(navigation)
        result.set_habitat_color(color)
        return result

    @staticmethod
    def _validate_navigation(navigation: np.ndarray) -> np.ndarray:
        value = np.asarray(navigation)
        if value.ndim != 2:
            raise ValueError("navigation must be a 2D array")
        return value

    @staticmethod
    def _validate_color(
            color: np.ndarray, shape: tuple[int, int]
        ) -> np.ndarray:
        value = np.asarray(color, dtype=np.uint8)
        if value.shape[:2] != shape or value.ndim != 3:
            raise ValueError(
                "color must match navigation height and width"
                )
        if value.shape[2] == 3:
            alpha = np.full((*shape, 1), 255, dtype=np.uint8)
            value = np.concatenate((value, alpha), axis=2)
        if value.shape[2] != 4:
            raise ValueError("color must contain RGB or RGBA channels")
        return value

    @staticmethod
    def habitat_to_graph_grid(array: np.ndarray) -> np.ndarray:
        """Convert Habitat [z, x] raster order to graph [x, y]."""
        return np.ascontiguousarray(np.asarray(array)[::-1, ::-1])

    def set_habitat_navigation(self, navigation: np.ndarray) -> None:
        navigation = self._validate_navigation(navigation)
        if navigation.shape != self.shape[:2]:
            raise ValueError("navigation shape does not match map shape")
        converted = self.habitat_to_graph_grid(navigation)
        self._grid[:, :, HABITAT_NAVIGATION_LAYER] = (converted
                                                        >= 1).astype(
                                                            np.float32
                                                            )

    def set_habitat_color(self, color: np.ndarray) -> None:
        color = self._validate_color(color, self.shape[:2])
        converted = self.habitat_to_graph_grid(color)
        packed = pack_rgba_batch(converted.reshape(-1, 4))
        self._grid[:, :,
                    HABITAT_COLOR_LAYER] = packed.reshape(self.shape[:2])

    def update_online_map(
            self,
            *,
            floor_points: np.ndarray | None = None,
            obstacle_points: np.ndarray | None = None
        ) -> None:
        if floor_points is not None and len(floor_points):
            self.update_points(
                np.asarray(floor_points),
                layer=ONLINE_OCCUPANCY_LAYER,
                value=1.0,
                )
        if obstacle_points is not None and len(obstacle_points):
            self.update_points(
                np.asarray(obstacle_points),
                layer=ONLINE_OCCUPANCY_LAYER,
                value=0.0,
                )

    def navigation_grid(self, *, copy: bool = False) -> np.ndarray:
        result = self._grid[:, :, HABITAT_NAVIGATION_LAYER]
        return result.copy() if copy else result

    def online_grid(self, *, copy: bool = False) -> np.ndarray:
        result = self._grid[:, :, ONLINE_OCCUPANCY_LAYER]
        return result.copy() if copy else result

    def color_rgba(self, *, copy: bool = False) -> np.ndarray:
        packed = np.ascontiguousarray(
            self._grid[:, :, HABITAT_COLOR_LAYER]
            )
        result = unpack_rgba_batch(packed.reshape(-1)
                                  ).reshape(*self.shape[:2], 4)
        return result.copy() if copy else result

    def snapshot(self) -> dict[str, Any]:
        return {
            "grid":
                self._grid.copy(),
            "origin":
                self.origin.copy(),
            "resolution":
                self.resolution,
            "floor_height":
                self.floor_height,
            "habitat_bounds": (
                None if self.habitat_bounds is None else
                self.habitat_bounds.copy()
                ),
            "graph_bounds": (
                None
                if self.graph_bounds is None else self.graph_bounds.copy()
                ),
            }

    def save(self, path: str | Path | None = None, to_bytes: bool = False):
        values = {
            "grid":
                self._grid,
            "origin":
                self.origin,
            "resolution":
                np.array(self.resolution, dtype=np.float64),
            "shape":
                np.asarray(self.shape[:2], dtype=np.int64),
            "floor_height":
                np.array(self.floor_height, dtype=np.float64),
            "habitat_bounds": (
                np.empty((0, 3), dtype=np.float32)
                if self.habitat_bounds is None else self.habitat_bounds
                ),
            "graph_bounds": (
                np.empty((0, 2), dtype=np.float32)
                if self.graph_bounds is None else self.graph_bounds
                ),
            }
        if to_bytes:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **values)
            return buffer.getvalue()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **values)

    @classmethod
    def load(
            cls,
            path: str | Path | None = None,
            bytes_data: bytes | None = None
        ) -> "TopdownMap2D":
        source = io.BytesIO(bytes_data) if bytes_data is not None else path
        with np.load(source) as data:
            habitat_bounds = data["habitat_bounds"]
            graph_bounds = data["graph_bounds"]
            result = cls(
                resolution=float(data["resolution"]),
                shape=tuple(int(value) for value in data["shape"]),
                origin=tuple(float(value) for value in data["origin"]),
                habitat_bounds=(
                    None if habitat_bounds.size == 0 else habitat_bounds
                    ),
                graph_bounds=(
                    None if graph_bounds.size == 0 else graph_bounds
                    ),
                floor_height=float(data["floor_height"]),
                )
            result._grid = np.asarray(data["grid"], dtype=np.float32)
        return result

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "TopdownMap2D":
        grid = np.asarray(snapshot["grid"], dtype=np.float32)
        result = cls(
            resolution=float(snapshot["resolution"]),
            shape=grid.shape[:2],
            origin=tuple(np.asarray(snapshot["origin"], dtype=float)),
            habitat_bounds=snapshot.get("habitat_bounds"),
            graph_bounds=snapshot.get("graph_bounds"),
            floor_height=float(snapshot.get("floor_height", 0.0)),
            )
        result._grid = grid.copy()
        return result
