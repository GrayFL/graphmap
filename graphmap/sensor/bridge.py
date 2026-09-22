"""Legacy rsim RGB + LiDAR bridge, retaining its existing public API."""
from typing import Optional, Tuple, Dict, Any, Union, Literal
import numpy as np
from ..pose import Pose
from ..visibility import frustum_culling
from ..misc.config import Munch, load_config
from .camera import CameraIntrinsic
from .sampling import sample_hw_data


class SensorBridge:
    """rsim RGB + LiDAR bridge; ``env.observe(**bridge.process_observation(obs))``.

    The legacy ``extrinsic_body_to_*`` names denote sensor poses IN body:
    T_body_camera / T_body_lidar map sensor points to body points. They are
    composed as T_world_body * T_body_sensor. New input sources should use
    graphmap.adapters instead of teaching this legacy bridge more formats.
    """

    def __init__(
        self,
        config: Munch,
        intrinsic_camera: CameraIntrinsic,
        extrinsic_body_to_camera: Optional[Pose] = None,
        extrinsic_body_to_lidar3d: Optional[Pose] = None,
        depth_range: Tuple[float, float] = (0.1, 100.0)
        ):
        """
        Parameters
        ----------
        intrinsic : CameraIntrinsic
            相机内参
        extrinsic_body_to_camera : Pose, optional
            T_body_camera，将 camera 点变换到 body；默认单位位姿
        extrinsic_body_to_lidar3d : Pose, optional
            T_body_lidar3d，将 lidar 点变换到 body；默认单位位姿
        depth_range : Tuple[float, float]
            (min_depth, max_depth), 深度裁剪范围
        """
        self.config = Munch.fromDict(config)
        self.intrinsic = intrinsic_camera
        self.ext_body2camera = (Pose(wrd_frame="body", ego_frame="camera")
                                if extrinsic_body_to_camera is None else extrinsic_body_to_camera)
        self.ext_body2lidar3d = (Pose(wrd_frame="body", ego_frame="lidar3d")
                                 if extrinsic_body_to_lidar3d is None else extrinsic_body_to_lidar3d)
        self.depth_min, self.depth_max = depth_range

    def colorize_points(
        self,
        pts_lidar: np.ndarray,
        sensor_pose: Pose,
        rgb_image: np.ndarray,
        intrinsic: CameraIntrinsic,
        default_color: Tuple[int, int, int] = (128, 128, 128)
        ) -> Munch:
        """将 LiDAR 点云变换到世界系并从 RGB 图像取色

        流程:
            1. LiDAR 点 → 世界系
            2. 世界点 → 相机系
            3. 相机系点 → 图像投影
            4. 从 RGB 图像采样颜色

        Parameters
        ----------
        pts_lidar : np.ndarray
            shape (N, 3) float32, LiDAR 坐标系点云
        sensor_pose : Pose
            传感器相对相机系位姿，例如 cam2lidar
        rgb_image : np.ndarray
            shape (H, W, 3) uint8, RGB 图像
        default_color : Tuple[int, int, int]
            不可见点的默认颜色

        Returns
        -------
        pts_world : np.ndarray
            shape (N, 3) float32, 世界系点云
        colors : np.ndarray
            shape (N, 3) or (N, 4) uint8, RGBA 颜色
        """
        N = pts_lidar.shape[0]
        img_h, img_w = rgb_image.shape[:2]

        pts_cam = sensor_pose(pts_lidar)
        pts_hwd = intrinsic.cam_to_hwd(pts_cam)
        h, w, d = pts_hwd[:, 0], pts_hwd[:, 1], pts_hwd[:, 2]

        # 4. 构建可见性 Mask (深度有效 且 在图像范围内)
        # 注意：HWD系统中，在相机前方的物体，其 d 是负数，物理深度为 -d
        valid_depth = (d < -self.depth_min)
        in_bounds = (h >= 0) & (h < img_h) & (w >= 0) & (w < img_w)
        visible_mask = valid_depth & in_bounds

        pts_hwd = pts_hwd[visible_mask]

        # 4. 采样颜色
        colors = np.full((N, 3), default_color, dtype=np.uint8)
        if np.any(visible_mask):
            vis_h = np.clip(h[visible_mask].astype(np.int32), 0, img_h - 1)
            vis_w = np.clip(w[visible_mask].astype(np.int32), 0, img_w - 1)

            # 从 RGB 图像中高级索引取色
            colors[visible_mask] = rgb_image[vis_h, vis_w, :3]
        return Munch(
            visible_mask=visible_mask,
            colors=colors,
            pts_hwd=pts_hwd,
            valid_depth=valid_depth,
            in_bounds=in_bounds,
            )

    def sample_points_from_hw_data(
        self,
        points: np.ndarray,
        T_pts2cam: Pose,
        hw_data: np.ndarray,
        *,
        intrinsic: CameraIntrinsic = None,
        default_value: Any = 0,
        index_mode: Literal["floor", "nearest"] = "nearest",
        ) -> Munch:
        """
        将任意坐标系下的一组空间点投影到图像，并从 H-W 数据中采样。

        Parameters
        ----------
        points : np.ndarray
            shape (N, 3)，某个源坐标系下的点。
        T_pts2cam : Pose
            必须满足：
                pts_cam = T_pts2cam(points)
            也就是说，它需要把 points 所在坐标系的点变换到相机坐标系。
        hw_data : np.ndarray
            shape (H, W) 或 (H, W, C...)。
            可以是 RGB 图、深度图、语义图、特征图等。
        intrinsic : CameraIntrinsic
            相机内参，提供 cam_to_hwd。
        default_value : Any
            不可见点的默认采样值。
        index_mode : {"floor", "nearest"}
            图像坐标到像素索引的转换方式。

        Returns
        -------
        Munch
            samples : np.ndarray
                shape (N,) 或 (N, C...)。
            visible_mask : np.ndarray
                shape (N,)。
            pts_cam : np.ndarray
                shape (N, 3)。
            pts_hwd : np.ndarray
                shape (N, 3)，完整的投影结果。
            visible_pts_hwd : np.ndarray
                shape (M, 3)，只包含可见点。
            valid_depth : np.ndarray
                shape (N,)。
            in_bounds : np.ndarray
                shape (N,)。
        """
        points = np.asarray(points)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points 必须是 shape (N, 3)，当前是 {points.shape}")

        pts_cam = T_pts2cam(points)
        intrinsic = intrinsic or self.intrinsic
        pts_hwd = intrinsic.cam_to_hwd(pts_cam)

        h = pts_hwd[:, 0]
        w = pts_hwd[:, 1]
        d = pts_hwd[:, 2]

        # 你的 HWD 约定：相机前方 d 为负数，物理深度为 -d
        valid_depth = d < -self.depth_min

        sampled = sample_hw_data(
            data=hw_data,
            h=h,
            w=w,
            valid_mask=valid_depth,
            default_value=default_value,
            index_mode=index_mode,
            )

        return Munch(
            samples=sampled.samples,
            visible_mask=sampled.visible_mask,
            pts_cam=pts_cam,
            pts_hwd=pts_hwd,
            visible_pts_hwd=pts_hwd[sampled.visible_mask],
            valid_depth=valid_depth,
            in_bounds=sampled.in_bounds,
            index_h=sampled.index_h,
            index_w=sampled.index_w,
            )

    # ------------------------------------------------------------------
    # 完整观测处理
    # ------------------------------------------------------------------

    def process_observation(
            self,
            obs_data: Dict[str, Any],
            pose_body: Optional[Pose] = None,
            rgb_image: Optional[np.ndarray] = None,
            pts_lidar: Optional[np.ndarray] = None
        ) -> Dict[str, Any]:
        """处理一次完整观测, 返回可直接传入 env.observe() 的参数字典

        支持两种输入方式:
        1. 从 JSON 字典自动提取: obs_data 需包含 'tf', 'camera', 'lidar3d' 键
        2. 显式传入: body_pose, rgb_image, pts_lidar (此时 obs_data 仅用 'tf')

        Parameters
        ----------
        obs_data : dict
            观测数据, 格式:
            {
                'tf': {'ret': [position (3,), quaternion (4,)]},
                'camera': {'ret': [[[r,g,b],...],...]},  # (H,W,3) nested list
                'lidar3d': {'ret': [[x,y,z],...]},       # (N,3) nested list
            }
        body_pose : Pose, optional
            机器人 body 在世界系的位姿 (从 tf 计算), 不传则从 obs_data['tf'] 构造
        rgb_image : np.ndarray, optional
            (H, W, 3) uint8 图像, 不传则从 obs_data['camera']['ret'] 构造
        pts_lidar : np.ndarray, optional
            (N, 3) float32, LiDAR 点云, 不传则从 obs_data['lidar3d']['ret'] 构造

        Returns
        -------
        dict
            {'pose_robot': Pose, 'points_camera': np.ndarray, 'colors': np.ndarray}
            可直接 ** 解包传入 env.observe()
        """
        timestamp = obs_data['tf']['timestamp']['data']

        # --- 解析 body 位姿 ---
        if pose_body is None:
            tf_data = obs_data['tf']['ret']
            pose_body = Pose(
                position=tf_data[0],
                rotation=tf_data[1],  # quaternion [x,y,z,w]
                label="body in world",
                )

        # --- 计算传感器位姿 ---
        pose_camera = pose_body * self.ext_body2camera
        pose_lidar3d = pose_body * self.ext_body2lidar3d

        # --- 解析 LiDAR 点云 ---
        if pts_lidar is None:
            pts_lidar = np.array(
                obs_data['lidar3d']['ret'], dtype=np.float32
                )

        pts_world = pose_lidar3d(pts_lidar)

        mask_floor = (
            (pts_world[:, 2] < self.config.map.clip_z[0])
            & (pts_world[:, 2] > self.config.map.under_floor_height)
            )
        pts_floor = pts_world[mask_floor]

        # mask_map_build = (
        #     (pts_world[:, 2] > self.config.map.clip_z[0])
        #     & (pts_world[:, 2] < self.config.map.clip_z[1])
        #     )  #
        # pts_world = pts_world[mask_map_build]
        # pts_lidar = pts_lidar[mask_map_build]

        # --- 解析 RGB 图像 ---
        if rgb_image is None:
            rgb_image = np.array(obs_data['camera']['ret'], dtype=np.uint8)

        # --- 颜色映射 & 世界变换 ---
        # cam2lidar = (~pose_camera) * pose_lidar3d
        cam2world = ~pose_camera
        ret = self.colorize_points(
            pts_world, cam2world, rgb_image, self.intrinsic
            )
        return {
            'pose_robot': pose_body,
            'pose_camera': pose_camera,
            'pose_lidar3d': pose_lidar3d,
            'points_world': pts_world,
            'points_floor': pts_floor,
            'points_hwd': ret.pts_hwd,
            'visible_mask': ret.visible_mask,
            'colors': ret.colors,
            'timestamp': timestamp,
            }

    def process_observation_world(
            self,
            pose_body: "Pose",
            pts_world: np.ndarray,
            rgb_image: np.ndarray,
            timestamp: Optional[float] = None
        ) -> Dict[str, Any]:
        """处理一次完整观测（世界坐标系点云），返回可直接传入 env.observe() 的参数字典。

        与 ``process_observation`` 的区别：
        - 点云已处于世界坐标系，跳过 lidar→world 变换。
        - 跳过雷达自身系的 frustum culling（点云已由外部注册系统处理）。

        Parameters
        ----------
        pose_body : Pose
            机器人 body 在世界系的位姿（来自 odometry）。
        pts_world : np.ndarray
            (N, 3) float32，世界坐标系中的 LiDAR 点云。
        rgb_image : np.ndarray
            (H, W, 3) uint8 图像。
        timestamp : float, optional
            观测时间戳（秒）。

        Returns
        -------
        dict
            与 ``process_observation`` 相同格式，可直接 ** 解包传入 env.observe()。
        """
        # --- 计算传感器位姿 ---
        pose_camera = pose_body * self.ext_body2camera
        pose_lidar3d = pose_body * self.ext_body2lidar3d

        pts_world = np.asarray(pts_world, dtype=np.float32)

        # --- 地面过滤 & Z 轴裁剪（与旧方法一致） ---
        mask_floor = (
            (pts_world[:, 2] < self.config.map.clip_z[0])
            & (pts_world[:, 2] > self.config.map.under_floor_height)
            )
        pts_floor = pts_world[mask_floor]

        mask_map_build = ((pts_world[:, 2] > self.config.map.clip_z[0])
                            &
                            (pts_world[:, 2] < self.config.map.clip_z[1]))
        pts_world = pts_world[mask_map_build]

        # --- 颜色映射 ---
        cam2world = ~pose_camera
        ret = self.colorize_points(
            pts_world, cam2world, rgb_image, self.intrinsic
            )
        return {
            'pose_robot': pose_body,
            'pose_camera': pose_camera,
            'pose_lidar3d': pose_lidar3d,
            'points_world': pts_world,
            'points_floor': pts_floor,
            'points_hwd': ret.pts_hwd,
            'visible_mask': ret.visible_mask,
            'colors': ret.colors,
            'timestamp': timestamp,
            }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def filter_by_frustum(
            self,
            pts_world: np.ndarray,
            camera_pose: Pose,
            fov_h=None,
            fov_v=None,
            near: float = 0.1,
            far: float = 100.0
        ) -> np.ndarray:
        """视锥体裁剪: 返回在世界系视锥内的点索引掩码"""
        return frustum_culling(
            pts_world, camera_pose, fov_h, fov_v, near, far
            )

    @classmethod
    def from_rsim_config(
            cls, config: Union[str, Dict, Munch]
        ) -> "SensorBridge":
        """从 rsim 兼容的 config (Munch / dict) 构造 SensorBridge

        自动映射:
            config.camera_params.rgb_sensor.fx/fy/ppx/ppy → CameraIntrinsic.from_sensor()
            config.camera_params.rgb_sensor.position/rotation → ext_body2cam
            config.camera_params.lidar3d_sensor.position/rotation → ext_body2lidar

        rotation 为 euler 角度 (degrees, xyz 顺序), 表示传感器在 body 系下的朝向.
        """
        from scipy.spatial.transform import Rotation as R

        if isinstance(config, str):
            config = load_config(config)
        elif isinstance(config, dict):
            config = Munch.fromDict(config)

        # 兼容 camera_params 和 sensor_params 两种 key
        cfg = config.get('camera_params'
                        ) or config.get('sensor_params') or config
        cfg_rgb = cfg.rgb_sensor
        cfg_lid = cfg.lidar3d_sensor

        intrinsic_camera = CameraIntrinsic.from_sensor(
            width=cfg_rgb.get("width"),
            height=cfg_rgb.get("height"),
            hfov=cfg_rgb.get("hfov"),
            vfov=cfg_rgb.get("vfov"),
            fx=cfg_rgb.fx,
            fy=cfg_rgb.fy,
            ppx=cfg_rgb.ppx,
            ppy=cfg_rgb.ppy,
            )

        ext_body2cam = Pose(
            position=np.array(cfg_rgb.position, dtype=np.float32),
            rotation=R.from_euler('xyz', cfg_rgb.rotation, degrees=True),
            label="body->camera"
            )
        ext_body2lid3d = Pose(
            position=np.array(cfg_lid.position, dtype=np.float32),
            rotation=R.from_euler('xyz', cfg_lid.rotation, degrees=True),
            label="body->lidar3d"
            )

        return cls(
            config=config,
            intrinsic_camera=intrinsic_camera,
            extrinsic_body_to_camera=ext_body2cam,
            extrinsic_body_to_lidar3d=ext_body2lid3d
            )


