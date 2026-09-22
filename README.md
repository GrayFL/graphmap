# graphmap

用于机器人点云处理与建图的 Python 工具库。可以单独使用位姿、坐标转换、相机内参、图像采样，也可以组合成 RGB-D / RGB + LiDAR → 带颜色及语义的全局点云 → 二维栅格的流程。

开发前先读 [PROJECT.md](PROJECT.md)。

**仅使用 Pose：** 其他项目需要位姿、外参或批量变换时，可直接阅读 [Pose 使用说明](graphmap/pose/README.md)，通过 `from graphmap.pose import Pose, PoseArray` 调用。

## 安装与环境

推荐本机 AIO：`/gray/LAB/envs/AIO/bin/python`。该环境已有 `.pth` 配置，在仓库外也能 `import graphmap`，无需修改环境。验证实际加载位置：

```bash
/gray/LAB/envs/AIO/bin/python -c 'import graphmap; print(graphmap.__file__)'
```

其他环境可在仓库根目录安装：

```bash
python -m pip install -e .                      # 几何、相机、3D 点云及语义地图
python -m pip install -e '.[mapping,config,dev]' # 加二维地图、图像 IO、YAML、测试与 notebook 执行
```

Python ≥ 3.11，NumPy ≥ 1.26.4，SciPy ≥ 1.17。`PoseArray` 使用 SciPy 1.17 引入的 N 维 Rotation 广播能力，不能只依据单个 `Pose` 能运行就降级 SciPy。[SciPy 发布说明](https://docs.scipy.org/doc/scipy/release/1.17.0-notes.html)

OpenCV 仅用于二维地图和数据集图片读取；YAML 按需加载。单独导入 `graphmap.pose` 不会加载地图、OpenCV 或绘图库。Notebook 绘图需要本地 `scipykit`，AIO 已提供，库本身不依赖它。

## 必须先了解的坐标约定

| 对象 | 约定 |
| --- | --- |
| 世界/机器人/graphmap 相机 | 右手系；局部 `X` 向前、`Y` 向左、`Z` 向上；世界 `Z` 向上，世界方向固定，不随机器人转动 |
| 位置、距离 | 米；输入深度的单位由适配器显式转换 |
| `Pose` | `T_world_camera` 把相机点变到世界：`p_world = T_world_camera(p_camera)` |
| 四元数 / 欧拉角 | `[x,y,z,w]`；欧拉角默认 `xyz`、**度**，弧度须传 `degrees=False` |
| 相机内参 FOV | **弧度**；`CameraIntrinsic.from_sensor()` 接收常见正焦距 `fx,fy` |
| 可见性函数 / 旧 Environment 配置 FOV | **度**；保留历史约定，传入内参 FOV 时须 `np.rad2deg(...)` |
| 图像 | `image[h,w]`；投影 `(h,w,d)` 中 `d=-X`，前方点 `d<0` |
| 二维地图 | `grid[h,w,layer]`，`h` 对应世界 `x`，`w` 对应世界 `y`；不转置 |
| 时间 | `observe(timestamp=...)` 接收会话内相对秒，内部为 int32 毫秒；不要直接传 Unix 秒 |

图像的 `h,w` 是像素行列，地图的 `h,w` 是世界 `x,y` 的栅格索引，两者不应混为同一三维坐标系。

## 1. 位姿、外参与批量点云

```python
import numpy as np
from graphmap.pose import Pose, PoseArray

T_world_body = Pose(x=1, yaw=90, wrd_frame="world", ego_frame="body")
T_body_camera = Pose(x=0.2, z=1, wrd_frame="body", ego_frame="camera")
T_world_camera = T_world_body * T_body_camera  # 右侧先执行

points_camera = np.array([[2., 0., 0.], [2., 0.5, 0.]])
points_world = T_world_camera(points_camera)
assert np.allclose((~T_world_camera)(points_world), points_camera)

trajectory = PoseArray.from_poses([Pose(), Pose(x=1), Pose(x=2, yaw=30)])
clouds_world = trajectory.apply_all(points_camera)  # (3, 2, 3)
```

`apply_each` 是逐点广播，`apply_clouds` 是每个位姿处理自己的点云，`apply_all` 是所有位姿处理同一份点云。`Pose(scale=...)` 支持正统一尺度 Sim(3)；传感器适配器要求米制刚体位姿，单位缩放应先作用于输入数据。

已有 OpenCV 或 Habitat 外参时，显式声明世界与局部坐标约定：

```python
from graphmap.pose.conventions import convert_points, convert_pose

points_graph = convert_points([[0.2, 0.3, 2.0]], source="opencv")
# [[2.0, -0.2, -0.3]]

T_world_optical = Pose(position=[1, 2, 3])  # 示例：光学相机到已有 graphmap 世界
T_world_camera = convert_pose(T_world_optical, world="graphmap", local="opencv")
```

`convert_pose` 实施 `C_world @ T @ inv(C_local)`。若输入是 world-to-camera，**先取逆**。若两个端点原本都是 Habitat 坐标，传 `world="habitat", local="habitat"`；已经转换过的位姿不要再次换轴。换轴不改变原点，也不会自动减去楼层高度。

## 2. RGB-D → 语义点云地图

以下示例不需要外部数据：

```python
import numpy as np
from graphmap.pose import Pose
from graphmap.sensor import CameraIntrinsic
from graphmap.adapters import rgbd_observation
from graphmap.environment import Environment

camera = CameraIntrinsic.from_sensor(
    width=64, height=48, fx=50, fy=50, ppx=32, ppy=24,
)
depth_mm = np.full((48, 64), 2000, dtype=np.uint16)
rgb = np.zeros((48, 64, 3), dtype=np.uint8)
rgb[..., 1] = 180
labels = np.full((48, 64), 7, dtype=np.uint32)

observation = rgbd_observation(
    depth_mm,
    intrinsic=camera,
    pose_camera=Pose(z=1),
    rgb=rgb,
    attributes={"semantic_id": labels},
    depth_scale=0.001,       # 原始数值乘以该系数得到米
    depth_kind="axial",     # 光轴深度；射线距离使用 "range"
    depth_range=(0.1, 10),
    stride=4,
    timestamp=0.0,
)
env = Environment(resolution=0.05)
env.observe(**observation)
print(env.status())
assert set(env.get_index_values("semantic_id")) == {7}
```

零、负、NaN、Inf 深度及可选 `mask` 排除的像素会同步从点、颜色、语义/特征中移除。`observation["pixel_indices"]` 保留每个输出点的原始 `[h,w]`，便于追踪来源。RGB 必须是 uint8 的 RGB/RGBA；深度、RGB 和属性图需要事先对齐、去畸变，分辨率一致。当前不自动处理鱼眼、畸变、时间同步或跨相机配准。

内参直接实现：

```text
h = pph + fh * Z/X       fh = -fy, pph = ppy
w = ppw + fw * Y/X       fw = -fx, ppw = ppx
d = -X
```

因此不用翻转 RGB 图片。`camera.cam_to_hwd(points)` 与 `camera.hwd_to_cam(hwd)` 可独立使用；正值深度图不能直接当作 `hwd` 的第三列。

## 3. 普通点云、LiDAR 与附加信息

```python
from graphmap.adapters import pointcloud_observation

observation = pointcloud_observation(
    points=[[1., 0., 0.], [2., 0., 0.]],
    pose_sensor=Pose(x=1),  # graphmap 局部传感器 → graphmap 世界
    attributes={"semantic_id": [10, 20], "confidence": [0.9, 0.7]},
    timestamp=1.0,
)
env.observe(**observation)
```

世界点云可传 `pose_sensor=Pose()`；真实机器人位姿可另外传 `pose_robot`。`attributes` 也支持逐点向量、字典等 IndexDB 可保存的值。适配器不猜测坐标约定：外部点云先用 `convert_points` 或已知外参变换到 graphmap。

`IndexDB` 有两类不同的键：

- 临时观测表：键是观测数组的**行号**，由适配器自动建立。
- `env.index_db("semantic_id")`：键是世界坐标与地图分辨率生成的 **uint64 体素键**，与数组当前行号无关。

环境按最近时间戳保留体素代表点；时间相同保留后来的点，并将胜出点的属性关联到体素键。多点共享同一属性值时只保存一份属性内容。查询使用 `env.get_index_values(name)` 或 `env.point_keys()`，不要把 `InfoPoints.index` 当作新的语义键，它是旧格式保留列。

旧 rsim RGB + LiDAR 流程仍可使用 `from graphmap.sensor import SensorBridge` 和 `SensorBridge.from_rsim_config(config)`。配置中的相机需提供 `width,height,fx,fy,ppx,ppy`。历史参数 `extrinsic_body_to_camera` 实际保存 **相机在 body 中的位姿** `T_body_camera`，作用方向为 camera → body。

## 4. 二维栅格与保存

```python
from graphmap.map2d import Map2D

grid = Map2D(resolution=0.1, shape=(200, 300), origin=(100, 150))
grid.update_from_occupancy(env.infopoints.xyz)
h, w = grid.world_to_grid([1.0, 2.0])
assert (h, w) == (110, 170)
grid[1.0, 2.0, 0] = 0  # 世界坐标索引
assert grid.grid[110, 170, 0] == 0
```

栅格值 `-1/0/1` 表示未知/障碍/可行。`world_to_grid` 使用 `np.round` 最近栅格，不是向下取整；`origin` 是世界零点的浮点栅格位置。绘制 `grid.grid[:, :, 0]` 时无需转置，横轴为 `w/y`、纵轴为 `h/x`。`Environment(enable_2d_map=True)` 可以自动标记观测点的占用，但不会自动完成地面分类或自由空间推断。

```python
env.save("outputs/demo_map")
restored = Environment.load("outputs/demo_map")
```

保存点云、命名属性表、可选二维地图和地图配置元数据。当前不保存机器人最新位姿、历史队列和完整外部配置。快照包含 pickle/NumPy object 元数据，只加载可信文件。

## Notebook 与验证

| Notebook | 内容 |
| --- | --- |
| [01_pose_and_frames.ipynb](examples/01_pose_and_frames.ipynb) | 外参链、批量位姿、OpenCV 换轴与三维绘图 |
| [02_rgbd_mapping.ipynb](examples/02_rgbd_mapping.ipynb) | 合成 RGB-D、语义/特征表、栅格与保存加载 |
| [03_hm3d_semantics.ipynb](examples/03_hm3d_semantics.ipynb) | 本地 HM3D 三帧 RGB-D 与语义融合，读取导出的相机位姿 |

选择 AIO 内核，从第一个单元格开始运行。所有绘图使用 `from scipykit.task.cv import *`，参考 [scipykit 本地 README](/gray/Storage/MyPyLib/scipykit/README.md) 以及 [mtp_initializer 绘图约定](/gray/Storage/MyPyLib/scipykit/mtp_initializer/README.md)。第三个示例默认读取 `/R/Datasets/3DGS/hm3dv2/0000`，可通过 `GRAPHMAP_HM3D_ROOT`、`GRAPHMAP_HM3D_SCENE` 修改；更换数据版本时需要核实 notebook 中明确列出的标定与导出约定。

```bash
/gray/LAB/envs/AIO/bin/python -m pytest -q
GRAPHMAP_HM3D_ROOT=/R/Datasets/3DGS/hm3dv2 /gray/LAB/envs/AIO/bin/python -m pytest -q
/gray/LAB/envs/AIO/bin/python scripts/run_notebooks.py
/gray/LAB/envs/AIO/bin/python scripts/run_notebooks.py --with-hm3d
```

普通测试不要求真实数据；设置数据集环境变量后才执行集成用例。Notebook 执行器使用启动它的 Python，结果写入临时目录或指定 `--output-dir`，不覆盖源 notebook。
