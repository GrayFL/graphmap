"""Sparse, footprint-aware ground navigation in metric graphmap coordinates.

The input is a snapshot of the fused world point cloud. Geometry is reassessed
when the robot moves; only explicit depth evidence of missing support persists.
No dataset IO, pose estimation, or sensor controller belongs in this module.
"""
from collections import deque
from dataclasses import asdict, dataclass
from enum import IntEnum
import heapq
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .pose import Pose


class GroundState(IntEnum):
    UNKNOWN = 0
    PRIOR = 1
    INCOMPLETE = 2
    DISCONNECTED = 3
    CONNECTED = 4
    OBSTACLE = 5
    DROP = 6
    STEP = 7
    SLOPE = 8
    ROUGH = 9


NAVIGATION_LAYERS = {"confidence": 0, "elevation": 1, "support": 2, "state": 3}


@dataclass(frozen=True)
class GroundConfig:
    """Robot capability and evidence thresholds; all distances are metres.

    Nodes default to half the smaller footprint side; support probes to 1/8.
    The rectangle is evaluated at the current robot yaw, including on edges.
    Confidence is an evidence score, not a calibrated probability.
    """
    footprint: tuple[float, float] = (0.8, 0.8)
    robot_height: float = 1.0
    max_step: float = 0.12
    max_slope_degrees: float = 15.0
    max_roughness: float = 0.035
    node_spacing: float | None = None
    support_spacing: float | None = None
    support_radius: float | None = None
    normal_radius: float = 0.18
    normal_neighbors: int = 16
    surface_roughness: float = 0.025
    merge_height: float = 0.05
    blind_radius: float = 1.0
    near_confidence: float = 0.6
    height_band: float = 0.5
    height_confidence: float = 0.3
    disconnected_confidence: float = 0.45
    depth_margin: float = 0.08

    def __post_init__(self):
        footprint = tuple(float(v) for v in self.footprint)
        if len(footprint) != 2 or not np.isfinite(footprint).all() or min(footprint) <= 0:
            raise ValueError("footprint must contain two finite positive side lengths")
        object.__setattr__(self, "footprint", footprint)
        for name, value in (("node_spacing", min(footprint) / 2),
                            ("support_spacing", min(footprint) / 8)):
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)
        if self.support_radius is None:
            object.__setattr__(self, "support_radius", self.support_spacing * 0.9)
        for name in ("robot_height", "max_step", "max_roughness", "node_spacing",
                     "support_spacing", "support_radius", "normal_radius", "surface_roughness",
                     "merge_height", "blind_radius", "height_band", "depth_margin"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.max_slope_degrees < 60:
            raise ValueError("max_slope_degrees must be between 0 and 60")
        if self.robot_height <= self.max_step:
            raise ValueError("robot_height must exceed max_step")
        if self.node_spacing > min(footprint) / 2 + 1e-9:
            raise ValueError("node_spacing must not exceed half the smaller footprint side")
        if self.support_spacing > min(footprint) / 4:
            raise ValueError("support_spacing must resolve at least four probes per footprint side")
        if (not isinstance(self.normal_neighbors, int) or isinstance(self.normal_neighbors, bool)
                or self.normal_neighbors < 6):
            raise ValueError("normal_neighbors must be an integer >= 6")
        for name in ("near_confidence", "height_confidence", "disconnected_confidence"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if self.near_confidence <= self.height_confidence:
            raise ValueError("near_confidence must exceed height_confidence")


def _points(value, name="points"):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite (N, 3) world coordinates")
    return value


def estimate_surface_normals(points, *, radius=0.18, neighbors=16):
    """Local PCA normals oriented towards +Z and RMS plane residuals.

    Unsupported/collinear neighborhoods return zero normals and infinite
    residual. These are geometric support candidates, not semantic floor IDs:
    tables and ceilings must still pass footprint and connectivity checks.
    """
    points = _points(points)
    if not np.isfinite(radius) or radius <= 0 or not isinstance(neighbors, int) or neighbors < 6:
        raise ValueError("radius must be positive and neighbors an integer >= 6")
    normals, residual = np.zeros_like(points), np.full(len(points), np.inf)
    if len(points) < 6:
        return normals, residual
    distances, ids = cKDTree(points).query(points, k=min(neighbors, len(points)))
    valid = distances <= radius
    count = valid.sum(axis=1)
    local = points[ids] - points[:, None, :]
    mean = (local * valid[..., None]).sum(axis=1) / count[:, None]
    centered = (local - mean[:, None, :]) * valid[..., None]
    covariance = np.einsum("nki,nkj->nij", centered, centered) / count[:, None, None]
    values, vectors = np.linalg.eigh(covariance)
    supported = (count >= 6) & (values[:, 1] > 1e-8)
    normals[supported] = vectors[supported, :, 0]
    normals[normals[:, 2] < 0] *= -1
    residual[supported] = np.sqrt(np.maximum(values[supported, 0], 0))
    return normals, residual


class GroundMap:
    """Sparse 3D support nodes, footprint evidence, and a navigation graph.

    ``update`` accepts the entire current cloud, not only the latest frame.
    ``ground_z`` explicitly supplies the support height at the robot's base;
    never use camera height as ground height. Multiple heights at the same XY
    remain separate nodes. The 2D raster is only a projection of this graph.
    """

    def __init__(self, config=None):
        self.config = GroundConfig() if config is None else config
        if not isinstance(self.config, GroundConfig):
            raise TypeError("config must be GroundConfig")
        self.xyz = np.empty((0, 3))
        self.normals = np.empty((0, 3))
        self.confidence = np.empty(0, dtype=np.float32)
        self.support = np.empty(0, dtype=np.float32)
        self.state = np.empty(0, dtype=np.uint8)
        self.edges = np.empty((0, 2), dtype=np.int64)
        self.edge_confidence = np.empty(0, dtype=np.float32)
        self.robot_position = np.zeros(3)
        self.ground_z = 0.0
        self.yaw = 0.0
        self.roots = np.empty(0, dtype=np.int64)
        self.robot_confidence = 0.0
        self.robot_support = 0.0
        self.robot_state = int(GroundState.UNKNOWN)
        self._voids = {}  # quantized [support x, support y, support z] -> world point
        self._targets = np.empty((0, 3))

    def _surfaces(self, points, normals=None):
        if normals is None:
            normals, residual = estimate_surface_normals(
                points, radius=self.config.normal_radius, neighbors=self.config.normal_neighbors)
        else:
            normals = _points(normals, "normals")
            if normals.shape != points.shape:
                raise ValueError("normals must align with points")
            norms = np.linalg.norm(normals, axis=1)
            normals = normals / np.maximum(norms[:, None], 1e-12)
            normals = normals * np.where(normals[:, 2:3] < 0, -1, 1)
            residual = np.zeros(len(points))
        # Keep steep support surfaces so they can be explicitly rejected as slope.
        mask = (normals[:, 2] >= 0.5) & (residual <= self.config.surface_roughness)
        return normals, mask

    def _make_nodes(self, surfaces, normals):
        cfg = self.config
        groups = {}
        for row, hw in enumerate(np.rint(surfaces[:, :2] / cfg.node_spacing).astype(np.int64)):
            groups.setdefault(tuple(hw), []).append(row)
        xyz, node_normals, cells = [], [], []
        for hw, rows in sorted(groups.items()):
            rows = np.asarray(rows)[np.argsort(surfaces[rows, 2])]
            splits = np.flatnonzero(np.diff(surfaces[rows, 2]) > cfg.merge_height) + 1
            for cluster in np.split(rows, splits):
                normal = np.mean(normals[cluster], axis=0)
                normal /= np.linalg.norm(normal)
                gradient = -normal[:2] / normal[2]
                xy = np.asarray(hw) * cfg.node_spacing
                z = np.median(surfaces[cluster, 2] + (xy - surfaces[cluster, :2]) @ gradient)
                xyz.append([*xy, z])
                node_normals.append(normal)
                cells.append(hw)

        # Hypotheses are local: one ring around measured, near-height support,
        # plus the current blind area. They never become measured floor points.
        hypotheses = set()
        for point, hw in zip(xyz, cells):
            if abs(point[2] - self.ground_z) <= cfg.height_band:
                hypotheses.update((hw[0] + a, hw[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1))
        robot_hw = np.rint(self.robot_position[:2] / cfg.node_spacing).astype(int)
        radius = int(np.ceil(cfg.blind_radius / cfg.node_spacing))
        for a in range(-radius, radius + 1):
            for b in range(-radius, radius + 1):
                hw = robot_hw + (a, b)
                if np.linalg.norm(hw * cfg.node_spacing - self.robot_position[:2]) <= cfg.blind_radius:
                    hypotheses.add(tuple(hw))
        present = {}
        for point, hw in zip(xyz, cells):
            present.setdefault(hw, []).append(point[2])
        for hw in sorted(hypotheses):
            if any(abs(z - self.ground_z) <= cfg.max_step for z in present.get(hw, [])):
                continue
            xyz.append([*(np.asarray(hw) * cfg.node_spacing), self.ground_z])
            node_normals.append([0, 0, 1])
        return np.asarray(xyz, dtype=float).reshape(-1, 3), np.asarray(node_normals, dtype=float).reshape(-1, 3)

    def _depth_evidence(self, queries, support_normals, depth, intrinsic, pose_camera, depth_scale, depth_kind, depth_range):
        """-1 unknown/occluded, 0 surface return, 1 ray passes expected support."""
        camera_points = (~pose_camera)(queries)
        forward = camera_points[:, 0]
        positive = forward > 1e-6
        pixels = np.zeros((len(queries), 2))
        pixels[positive] = intrinsic.cam_to_hwd(camera_points[positive])[:, :2]
        hw = np.floor(pixels).astype(np.int64)
        inside = positive & (hw[:, 0] >= 0) & (hw[:, 0] < intrinsic.height - 1)
        inside &= (hw[:, 1] >= 0) & (hw[:, 1] < intrinsic.width - 1)
        evidence = np.full(len(queries), -1, dtype=np.int8)
        rows = np.flatnonzero(inside)
        h, w = hw[rows].T
        readings = np.column_stack([depth[h + dh, w + dw] for dh, dw in ((0, 0), (0, 1), (1, 0), (1, 1))]) * depth_scale
        valid = np.isfinite(readings).all(axis=1) & (readings > depth_range[0]).all(axis=1)
        valid &= (readings < depth_range[1]).all(axis=1)
        expected = forward[rows] if depth_kind == "axial" else np.linalg.norm(camera_points[rows], axis=1)
        valid &= (expected > depth_range[0]) & (expected < depth_range[1])
        world_ray = (queries[rows] - pose_camera.position) / expected[:, None]
        incidence = np.abs(np.sum(world_ray * support_normals[rows], axis=1))
        # Plane-height uncertainty expands along grazing rays; a small floor
        # residual must not look like a hole merely because the camera is far.
        margin = self.config.depth_margin + self.config.surface_roughness / np.maximum(incidence, 1e-6)
        # All four pixels must see beyond the patch; mixed depth boundaries and
        # invalid pixels cannot prove a hole. A nearer surface is occlusion.
        passed = valid & (readings.min(axis=1) > expected + margin)
        hit = valid & (np.max(np.abs(readings - expected[:, None]), axis=1) <= margin)
        evidence[rows[passed]], evidence[rows[hit]] = 1, 0
        return evidence

    def _void_keys(self, queries):
        scales = [self.config.support_spacing, self.config.support_spacing, self.config.merge_height]
        return np.rint(queries / scales).astype(np.int64)

    def _evaluate(self, centers, normals, scene_tree, points, surface_tree, surfaces,
                  surface_xy_tree, depth_args, *, footprint=None):
        cfg = self.config
        footprint = np.asarray(cfg.footprint if footprint is None else footprint)
        nx, ny = [int(np.ceil(side / cfg.support_spacing)) + 1 for side in footprint]
        a, b = np.meshgrid(np.linspace(-footprint[0] / 2, footprint[0] / 2, nx),
                           np.linspace(-footprint[1] / 2, footprint[1] / 2, ny), indexing="ij")
        rotation = np.array([[np.cos(self.yaw), -np.sin(self.yaw)], [np.sin(self.yaw), np.cos(self.yaw)]])
        offsets = np.column_stack([a.ravel(), b.ravel()]) @ rotation.T
        slopes = -normals[:, :2] / normals[:, 2:3]
        queries = np.empty((len(centers), len(offsets), 3))
        queries[:, :, :2] = centers[:, None, :2] + offsets
        queries[:, :, 2] = centers[:, 2:3] + slopes @ offsets.T
        flat = queries.reshape(-1, 3)
        supported = np.zeros(len(flat), dtype=bool)
        chosen_z = flat[:, 2].copy()
        lower, higher = np.zeros(len(flat), bool), np.zeros(len(flat), bool)
        if len(surfaces):
            radius = np.hypot(cfg.support_radius, cfg.max_step + cfg.max_roughness)
            distances, ids = surface_tree.query(flat, k=4, distance_upper_bound=radius)
            candidate = surfaces[np.minimum(ids, len(surfaces) - 1)]
            xy_distance = np.linalg.norm(candidate[:, :, :2] - flat[:, None, :2], axis=2)
            suitable = np.isfinite(distances) & (xy_distance <= cfg.support_radius)
            scores = np.where(suitable, distances, np.inf)
            best = np.argmin(scores, axis=1)
            supported = np.isfinite(scores[np.arange(len(flat)), best])
            chosen_z[supported] = candidate[np.arange(len(flat)), best, 2][supported]
            xy_distance, nearest = surface_xy_tree.query(flat[:, :2])
            delta_z = surfaces[nearest, 2] - flat[:, 2]
            nearby = ~supported & (xy_distance <= cfg.support_radius)
            lower = nearby & (delta_z < -cfg.max_step - cfg.max_roughness)
            higher = nearby & (delta_z > cfg.max_step + cfg.max_roughness) & (delta_z < cfg.robot_height)

        expected_support = flat.copy()
        expected_support[supported, 2] = chosen_z[supported]
        keys = self._void_keys(expected_support)
        if depth_args is not None:
            evidence = self._depth_evidence(expected_support, np.repeat(normals, len(offsets), axis=0), **depth_args)
            for row in np.flatnonzero(evidence == 0):
                self._voids.pop(tuple(keys[row]), None)
            for row in np.flatnonzero(evidence == 1):
                self._voids[tuple(keys[row])] = expected_support[row].copy()
        void = np.fromiter((tuple(key) in self._voids for key in keys), bool, len(keys))
        supported &= ~void
        support = supported.reshape(len(centers), -1)
        floor_z = chosen_z.reshape(len(centers), -1)
        fraction = support.mean(axis=1)
        state = np.full(len(centers), GroundState.INCOMPLETE, dtype=np.uint8)
        full = support.all(axis=1)
        state[full] = GroundState.DISCONNECTED
        slope_bad = np.linalg.norm(slopes, axis=1) > np.tan(np.deg2rad(cfg.max_slope_degrees))
        state[slope_bad] = GroundState.SLOPE

        # Check measured local relief, including discontinuities inside a robot
        # footprint. A room's absolute elevation is not a collision criterion.
        design = np.column_stack([offsets, np.ones(len(offsets))])
        neighbors = scene_tree.query_ball_point(centers[:, :2], np.linalg.norm(footprint) / 2 + 1e-8) if len(points) else [[] for _ in centers]
        for i, rows in enumerate(neighbors):
            valid = support[i]
            if valid.sum() >= 6 and np.linalg.matrix_rank(design[valid]) == 3:
                fitted, *_ = np.linalg.lstsq(design[valid], floor_z[i, valid], rcond=None)
                roughness = np.sqrt(np.mean((design[valid] @ fitted - floor_z[i, valid])**2))
                if np.linalg.norm(fitted[:2]) > np.tan(np.deg2rad(cfg.max_slope_degrees)) + 1e-6:
                    state[i] = GroundState.SLOPE
                if roughness > cfg.max_roughness:
                    state[i] = GroundState.ROUGH
            z_patch, known = floor_z[i].reshape(nx, ny), valid.reshape(nx, ny)
            for axis in (0, 1):
                pair = (np.take(known, range(known.shape[axis] - 1), axis=axis)
                        & np.take(known, range(1, known.shape[axis]), axis=axis))
                if np.any(np.abs(np.diff(z_patch, axis=axis))[pair] > cfg.max_step + 1e-6):
                    state[i] = GroundState.STEP
            nearby = points[rows]
            delta = nearby[:, :2] - centers[i, :2]
            local = delta @ rotation
            inside = (np.abs(local) <= footprint / 2 + 1e-8).all(axis=1)
            relative_z = nearby[:, 2] - centers[i, 2] - delta @ slopes[i]
            # A permitted low step can lift the body under an overhang. Include
            # the highest measured support residual in the clearance envelope.
            lift = max(0., float(np.max(floor_z[i, valid] - queries[i, valid, 2]))) if valid.any() else 0.
            low_obstacles = relative_z[inside & (relative_z > 0) & (relative_z <= cfg.max_step)]
            if len(low_obstacles):
                lift = max(lift, float(low_obstacles.max()))
            if np.any(inside & (relative_z > cfg.max_step + 1e-6)
                      & (relative_z <= cfg.robot_height + lift + 1e-6)):
                state[i] = GroundState.OBSTACLE
        state[higher.reshape(len(centers), -1).any(axis=1)] = GroundState.STEP
        state[(lower | void).reshape(len(centers), -1).any(axis=1)] = GroundState.DROP
        blocked = state >= GroundState.OBSTACLE
        full &= ~blocked
        near_height = np.abs(centers[:, 2] - self.ground_z) <= cfg.height_band
        confidence = np.where(near_height, cfg.height_confidence, 0.1)
        confidence[full & near_height] = cfg.disconnected_confidence
        distance = np.linalg.norm(centers[:, :2] - self.robot_position[:2], axis=1)
        prior = cfg.height_confidence + (cfg.near_confidence - cfg.height_confidence) * np.maximum(0, 1 - distance / cfg.blind_radius)
        confidence[near_height] = np.maximum(confidence[near_height], prior[near_height])
        confidence[blocked] = 0
        state[(fraction == 0) & ~blocked] = GroundState.PRIOR
        targets = queries[~support & ~blocked[:, None]]
        return confidence, fraction, state, full, targets

    def update(self, points, *, robot_pose, ground_z, normals=None, depth=None,
               intrinsic=None, pose_camera=None, depth_scale=1.0, depth_kind="axial",
               depth_range=(0.0, np.inf)):
        """Reassess a fused cloud; optionally add current RGB-D support evidence.

        Depth zero/NaN/invalid pixels and occluded expected floor stay unknown.
        A valid return behind expected support is negative evidence; a later
        return on that support clears it. Invalid depth never clears a hole.
        """
        points = _points(points)
        if not isinstance(robot_pose, Pose) or not robot_pose.is_rigid or not np.isfinite(ground_z):
            raise ValueError("robot_pose must be rigid and ground_z finite")
        depth_args = None
        if depth is not None:
            depth = np.asarray(depth, dtype=float)
            if intrinsic is None or depth.shape != (intrinsic.height, intrinsic.width):
                raise ValueError("depth shape must match intrinsic")
            if not isinstance(pose_camera, Pose) or not pose_camera.is_rigid:
                raise ValueError("depth evidence requires a rigid pose_camera")
            if not np.isfinite(depth_scale) or depth_scale <= 0 or depth_kind not in {"axial", "range"}:
                raise ValueError("invalid depth_scale or depth_kind")
            near, far = depth_range
            if not np.isfinite(near) or near < 0 or np.isnan(far) or far <= near:
                raise ValueError("depth_range must satisfy 0 <= near < far")
            depth_args = dict(depth=depth, intrinsic=intrinsic, pose_camera=pose_camera,
                              depth_scale=depth_scale, depth_kind=depth_kind, depth_range=depth_range)
        all_normals, mask = self._surfaces(points, normals)
        surfaces = points[mask]
        self.robot_position = np.asarray(robot_pose.position, dtype=float).copy()
        self.ground_z = float(ground_z)
        forward = robot_pose.rotation.apply([1., 0., 0.])
        self.yaw = float(np.arctan2(forward[1], forward[0]))
        self.xyz, self.normals = self._make_nodes(surfaces, all_normals[mask])
        geometry = (cKDTree(points[:, :2]), points, cKDTree(surfaces), surfaces,
                    cKDTree(surfaces[:, :2]), depth_args)
        conf, self.support, self.state, full, self._targets = self._evaluate(self.xyz, self.normals, *geometry)

        # Four-neighbor spatial edges, preserving separate surface heights.
        groups = {}
        cells = np.rint(self.xyz[:, :2] / self.config.node_spacing).astype(int)
        for i, cell in enumerate(cells):
            groups.setdefault(tuple(cell), []).append(i)
        pairs = []
        max_delta = self.config.max_step + self.config.node_spacing * np.tan(np.deg2rad(self.config.max_slope_degrees))
        for i, (h, w) in enumerate(cells):
            if conf[i] == 0:
                continue
            for neighbor in ((h + 1, w), (h, w + 1)):
                for j in groups.get(neighbor, []):
                    if conf[j] > 0 and abs(self.xyz[i, 2] - self.xyz[j, 2]) <= max_delta:
                        pairs.append((i, j))
        self.edges = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if len(self.edges):
            midpoints = self.xyz[self.edges].mean(axis=1)
            midnormals = self.normals[self.edges].mean(axis=1)
            midconf, midfull = np.zeros(len(self.edges)), np.zeros(len(self.edges), bool)
            movement = self.xyz[self.edges[:, 1], :2] - self.xyz[self.edges[:, 0], :2]
            rotation = np.array([[np.cos(self.yaw), -np.sin(self.yaw)], [np.sin(self.yaw), np.cos(self.yaw)]])
            for axis in (0, 1):
                rows = np.flatnonzero(np.abs(movement[:, axis]) > self.config.node_spacing / 2)
                if not len(rows):
                    continue
                # A local rectangle enclosing the ENTIRE translating footprint,
                # not merely its endpoints/midpoint. Conservative at oblique yaw.
                swept = np.asarray(self.config.footprint) + np.abs(movement[rows[0]] @ rotation)
                midconf[rows], _, _, midfull[rows], _ = self._evaluate(
                    midpoints[rows], midnormals[rows], *geometry, footprint=swept)
        else:
            midconf, midfull = np.empty(0), np.empty(0, bool)
        confirmed_edges = midfull & full[self.edges].all(axis=1)
        adjacency = [[] for _ in self.xyz]
        for (i, j) in self.edges[confirmed_edges]:
            adjacency[i].append(j)
            adjacency[j].append(i)
        distance = np.linalg.norm(self.xyz[:, :2] - self.robot_position[:2], axis=1)
        anchors = full & (distance <= self.config.node_spacing * 0.75)
        anchors &= np.abs(self.xyz[:, 2] - self.ground_z) <= self.config.max_step
        robot_center = np.array([[*self.robot_position[:2], self.ground_z]])
        nearest = np.argmin(np.linalg.norm(self.xyz - robot_center, axis=1))
        robot_normal = self.normals[nearest:nearest + 1]
        rconf, rsupport, rstate, rfull, _ = self._evaluate(robot_center, robot_normal, *geometry)
        self.robot_confidence = 1.0 if rfull[0] else float(rconf[0])
        self.robot_support = float(rsupport[0])
        self.robot_state = int(GroundState.CONNECTED if rfull[0] else rstate[0])
        if not rfull[0]:
            anchors[:] = False
        else:
            # A free snapped node does not prove that the actual robot stance
            # or its connection to that node is free.
            rotation = np.array([[np.cos(self.yaw), -np.sin(self.yaw)], [np.sin(self.yaw), np.cos(self.yaw)]])
            for row in np.flatnonzero(anchors):
                swept = np.asarray(self.config.footprint) + np.abs((self.xyz[row, :2] - robot_center[0, :2]) @ rotation)
                _, _, _, valid, _ = self._evaluate(
                    (self.xyz[row:row + 1] + robot_center) / 2,
                    (self.normals[row:row + 1] + robot_normal) / 2,
                    *geometry, footprint=swept)
                anchors[row] = valid[0]
        self.roots = np.flatnonzero(anchors)
        connected = anchors.copy()
        queue = deque(np.flatnonzero(anchors))
        while queue:
            for neighbor in adjacency[queue.popleft()]:
                if not connected[neighbor]:
                    connected[neighbor] = True
                    queue.append(neighbor)
        conf[connected], self.state[connected] = 1.0, GroundState.CONNECTED
        self.confidence = conf.astype(np.float32)
        self.support = self.support.astype(np.float32)
        self.edge_confidence = np.minimum(conf[self.edges].min(axis=1), np.where(midfull, 1., midconf)).astype(np.float32)
        return self

    def point_attributes(self, points, *, normals=None):
        """Label actual cloud rows; never insert hypothetical support into 3D.

        Non-support surfaces get zero confidence. ``ground_candidate`` only
        denotes local support-like geometry; ``ground_state`` explains why a
        candidate is unconfirmed or blocked. Recompute all rows after update.
        """
        points = _points(points)
        _, candidate = self._surfaces(points, normals)
        confidence = np.zeros(len(points), dtype=np.float32)
        state = np.full(len(points), GroundState.UNKNOWN, dtype=np.uint8)
        if len(self.xyz) and candidate.any():
            rows = np.flatnonzero(candidate)
            _, nearest = cKDTree(self.xyz).query(points[rows])
            delta = points[rows] - self.xyz[nearest]
            gradient = -self.normals[nearest, :2] / self.normals[nearest, 2:3]
            aligned = np.abs(delta[:, 2] - np.sum(delta[:, :2] * gradient, axis=1)) <= self.config.max_step
            aligned &= np.linalg.norm(delta[:, :2], axis=1) <= self.config.node_spacing * 0.8
            confidence[rows[aligned]] = self.confidence[nearest[aligned]]
            state[rows[aligned]] = self.state[nearest[aligned]]
        return {"ground_confidence": confidence, "ground_state": state, "ground_candidate": candidate}

    def to_map2d(self):
        """Rasterize confidence/elevation/support/state; h -> x, w -> y.

        Prefer a confirmed connected layer; otherwise select the layer nearest
        current ground height. Overlapping floors remain distinct in ``xyz``
        and the graph. Unknown raster cells have confidence -1, other layers NaN.
        """
        from .map2d import Map2D
        if not len(self.xyz):
            raise ValueError("update the ground map before rasterizing")
        cells = np.rint(self.xyz[:, :2] / self.config.node_spacing).astype(int)
        lower, upper = cells.min(axis=0) - 1, cells.max(axis=0) + 1
        shape = upper - lower + 1
        if np.prod(shape, dtype=np.float64) > 4_000_000:
            raise ValueError("navigation raster exceeds four million cells; use the sparse graph")
        result = Map2D(resolution=self.config.node_spacing, shape=tuple(shape), origin=tuple(-lower), num_layers=4)
        result.grid[:] = np.nan
        result.grid[:, :, 0] = -1
        selected = {}
        for i, cell in enumerate(cells):
            score = (self.confidence[i] == 1, -abs(self.xyz[i, 2] - self.ground_z))
            key = tuple(cell)
            if key not in selected or score > selected[key][0]:
                selected[key] = (score, i)
        for cell, (_, i) in selected.items():
            h, w = np.asarray(cell) - lower
            result.grid[h, w] = [self.confidence[i], self.xyz[i, 2], self.support[i], self.state[i]]
        return result

    def plan_path(self, start, goal, *, min_confidence=1.0):
        """Dijkstra on checked footprint edges at this map's fixed yaw.

        Endpoints are graph support locations in XYZ; snapping is limited to
        0.8 node spacing and max_step in height. Default paths use only measured
        connected support. A lower threshold explicitly permits uncertain terrain.
        Returns an empty (0,3) array when no such route exists.
        """
        endpoints = _points([start, goal], "start/goal")
        if not 0 < min_confidence <= 1:
            raise ValueError("min_confidence must be in (0, 1]")
        if not len(self.xyz):
            return np.empty((0, 3))
        snapped = []
        for point in endpoints:
            delta = self.xyz - point
            allowed = (np.linalg.norm(delta[:, :2], axis=1) <= self.config.node_spacing * .8)
            allowed &= (np.abs(delta[:, 2]) <= self.config.max_step) & (self.confidence >= min_confidence)
            if not allowed.any():
                return np.empty((0, 3))
            rows = np.flatnonzero(allowed)
            snapped.append(int(rows[np.argmin(np.linalg.norm(delta[rows], axis=1))]))
        start_id, goal_id = snapped
        adjacency = [[] for _ in self.xyz]
        for (i, j), confidence in zip(self.edges, self.edge_confidence):
            if confidence >= min_confidence:
                cost = np.linalg.norm(self.xyz[i] - self.xyz[j]) / confidence
                adjacency[i].append((j, cost))
                adjacency[j].append((i, cost))
        costs, previous, queue = {start_id: 0.}, {}, [(0., start_id)]
        while queue:
            cost, row = heapq.heappop(queue)
            if cost > costs[row]:
                continue
            if row == goal_id:
                path = [row]
                while row in previous:
                    row = previous[row]
                    path.append(row)
                return self.xyz[path[::-1]].copy()
            for neighbor, edge_cost in adjacency[row]:
                next_cost = cost + edge_cost
                if next_cost < costs.get(neighbor, np.inf):
                    costs[neighbor], previous[neighbor] = next_cost, row
                    heapq.heappush(queue, (next_cost, neighbor))
        return np.empty((0, 3))

    def observation_targets(self, *, limit=12, max_distance=5.0):
        """Expected support patches to observe next, ordered from near to far.

        These are world look-at targets, not executable camera/motion commands.
        Unknown or occluded patches need another view; targets don't raise confidence.
        """
        if not isinstance(limit, int) or limit < 1 or not np.isfinite(max_distance) or max_distance <= 0:
            raise ValueError("limit and max_distance must be positive")
        distance = np.linalg.norm(self._targets[:, :2] - self.robot_position[:2], axis=1)
        result = []
        for row in np.argsort(distance):
            if distance[row] > max_distance:
                break
            point = self._targets[row]
            if not result or np.min(np.linalg.norm(np.asarray(result)[:, :2] - point[:2], axis=1)) >= self.config.node_spacing:
                result.append(point)
                if len(result) == limit:
                    break
        return np.asarray(result).reshape(-1, 3)

    def save(self, path):
        """Save graph, parameters and explicit missing-support evidence (no pickle)."""
        np.savez_compressed(path, config=json.dumps(asdict(self.config)), version=1,
                            xyz=self.xyz, normals=self.normals, confidence=self.confidence,
                            support=self.support, state=self.state, edges=self.edges,
                            edge_confidence=self.edge_confidence, robot_position=self.robot_position,
                            roots=self.roots, robot_confidence=self.robot_confidence,
                            robot_support=self.robot_support, robot_state=self.robot_state,
                            ground_z=self.ground_z, yaw=self.yaw, targets=self._targets,
                            voids=np.asarray(list(self._voids.values())).reshape(-1, 3))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as saved:
            if int(saved["version"]) != 1:
                raise ValueError("unsupported GroundMap version")
            instance = cls(GroundConfig(**json.loads(str(saved["config"]))))
            for name in ("xyz", "normals", "confidence", "support", "state", "edges", "edge_confidence", "robot_position", "roots"):
                setattr(instance, name, saved[name].copy())
            instance.robot_confidence = float(saved["robot_confidence"])
            instance.robot_support = float(saved["robot_support"])
            instance.robot_state = int(saved["robot_state"])
            instance.ground_z, instance.yaw = float(saved["ground_z"]), float(saved["yaw"])
            instance._targets = saved["targets"].copy()
            voids = saved["voids"]
            instance._voids = {tuple(key): point.copy() for key, point in zip(instance._void_keys(voids), voids)}
        return instance
