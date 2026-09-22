"""Estimate LiDAR ray directions independently of cameras and mapping."""
import numpy as np
from scipy.spatial import cKDTree


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


