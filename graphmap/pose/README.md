# Pose v3 使用说明

安装 graphmap 的方式见 [总 README](../../README.md#安装与环境)。其他项目可直接通过 `from graphmap.pose import Pose, PoseArray` 使用本模块，本文集中说明位姿与变换接口。

`graphmap.pose` 提供统一的单个位姿、批量位姿和相似变换接口：

- `Pose`：单个 SE(3) 位姿；当 `scale != 1` 时也可表示方向保持的 Sim(3) 变换。
- `PoseArray`：具有 NumPy 广播语义的 N 维位姿数组。
- `Transform = Pose`：语义别名，适合强调“这是一个操作其他对象的变换”。
- `Sim3 = Similarity3 = Pose`：兼容且更明确的 Sim(3) 别名。
- `TransformArray = PoseArray`：批量变换别名。

依赖：

```bash
pip install 'numpy>=1.26.4' 'scipy>=1.17'
```

## 1. 约定

对象保存 `(scale, rotation, translation)`，对点执行主动变换：

```text
y = translation + scale * rotation(x)
```

坐标系标签 `Pose(wrd_frame="world", ego_frame="camera")` 表示：

```text
T_world_camera: camera -> world
```

四元数采用 SciPy 顺序 `[x, y, z, w]`。欧拉角默认为 `xyz` 顺序和角度制。

## 2. 创建单个位姿

### 欧拉角

长度为 3 的 `rotation` 默认按角度制欧拉角解析：

```python
from graphmap.pose import Pose

T_world_camera = Pose(
    translation=[1.0, 2.0, 3.0],
    rotation=[10.0, 20.0, 30.0],
    wrd_frame="world",
    ego_frame="camera",
)
```

也可以使用旧式分量参数：

```python
T = Pose(x=1, y=2, z=3, roll=10, pitch=20, yaw=30)
```

弧度欧拉角：

```python
T = Pose(rotation=[0.0, 0.0, 1.57], degrees=False)
```

三维旋转向量必须明确声明：

```python
T = Pose(rotation=[0.0, 0.0, 1.57], rotation_type="rotvec")
```

### 旋转矩阵、四元数和 SciPy Rotation

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from graphmap.pose import Pose

T_matrix = Pose(rotation=np.eye(3))
T_quat = Pose(rotation=[0.0, 0.0, 0.0, 1.0])
T_scipy = Pose(rotation=R.from_euler("z", 45, degrees=True))
```

`position` 与 `translation` 是构造参数别名；实例上的 `position`、`translation` 和 `t` 也指向相同平移量。

### 从矩阵创建

```python
T2 = Pose.from_matrix(T_world_camera.matrix)
```

矩阵左上角允许为 `scale * R`，因此 SE(3) 和正尺度 Sim(3) 都可恢复。

## 3. 创建 PoseArray

```python
import numpy as np
from graphmap.pose import PoseArray

positions = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
])

eulers_deg = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 30.0],
    [0.0, 0.0, 60.0],
])

poses = PoseArray(
    position=positions,
    rotation=eulers_deg,
    wrd_frame="world",
    ego_frame="camera",
)

print(poses.shape)       # (3,)
print(poses.position)    # (3, 3)
print(poses.quat)        # (3, 4)
print(poses.matrix)      # (3, 4, 4)
```

单个平移会被提升为 `batch=1`：

```python
one = PoseArray(position=[0, 0, 0], rotation=np.eye(3))
print(one.shape)  # (1,)
```

### `(3, 3)` 输入的判定

`(3, 3)` 既可能是一个旋转矩阵，也可能是 3 组欧拉角。自动规则会参考平移的批量形状：

```python
# position 的 batch shape 为 (3,)，所以 rotation 被解释为 3 组欧拉角
poses = PoseArray(
    position=np.zeros((3, 3)),
    rotation=np.zeros((3, 3)),
)

# position 被提升为 batch shape (1,)，所以 rotation 被解释为单个矩阵
one = PoseArray(
    position=[0, 0, 0],
    rotation=np.eye(3),
)
```

需要覆盖自动规则时显式指定：

```python
poses = PoseArray(
    position=np.zeros((3, 3)),
    rotation=np.eye(3),
    rotation_type="matrix",
)
```

## 4. 对点应用变换

### 单个位姿

```python
points_world = T_world_camera.apply(points_camera)
points_camera = T_world_camera.apply(points_world, inverse=True)

# 等价别名
points_world = T_world_camera.ego_to_wrd(points_camera)
points_camera = T_world_camera.wrd_to_ego(points_world)
points_world = T_world_camera(points_camera)
```

输入可以是 `(3,)`、`(N, 3)` 或任意 `(..., 3)`，前导形状会保留。

### 每个位姿对应一个点

```python
points = np.random.randn(3, 3)
world_points = poses.apply_each(points)  # (3, 3)
```

### 每个位姿对应自己的点云

```python
clouds = np.random.randn(3, 100, 3)
world_clouds = poses.apply_clouds(clouds)  # (3, 100, 3)
```

### 所有位姿作用于同一份点云

```python
shared_cloud = np.random.randn(100, 3)
all_clouds = poses.apply_all(shared_cloud)  # (3, 100, 3)
```

以上操作均为数组计算，不会拆成 Python `Pose` 列表逐个执行。

## 5. 位姿群复合

`*` 始终表示精确的 SE(3)/Sim(3) 群复合：

```python
T_world_camera = T_world_lidar * T_lidar_camera
T_camera_world = ~T_world_camera
relative = T_world_camera / T_world_lidar
```

对于 Sim(3)：

```text
(s1, R1, t1) * (s2, R2, t2)
= (s1*s2, R1*R2, t1 + s1*R1(t2))
```

`PoseArray` 与 `Pose`、`PoseArray` 之间会按 NumPy 规则广播复合。

## 6. 创建 Transform / Sim3

`Transform` 与 `Pose` 是同一个类，只是名称更适合作为操作符：

```python
from graphmap.pose import Transform

G = Transform(
    translation=[1.0, 0.0, 0.0],
    rotation=[0.0, 0.0, 30.0],
    scale=1.5,
)
```

绕指定点创建变换：

```python
G_about = Transform.from_about(
    rotation=[0.0, 0.0, 90.0],
    scale=1.2,
    about=[10.0, 0.0, 0.0],
    translation=[1.0, 2.0, 0.0],
)
```

它对点执行：

```text
y = about + translation + scale * R(x - about)
```

## 7. 对位姿应用通用 transform

```python
result = pose.transform(G, mode="world", about=center)
```

设待变换位姿为 `(s_p, R_p, p)`，操作符为 `(s, Q, d)`。

| mode | 位移 | 朝向 | 位姿自身 scale | 典型用途 |
|---|---|---|---|---|
| `world` | `p' = c+d+sQ(p-c)` | `R' = Q R_p` | 保持 `s_p` | Sim(3) 轨迹对齐、整体世界系变换 |
| `position` | `p' = c+d+sQ(p-c)` | 保持 `R_p` | 保持 `s_p` | 绕点公转但保持绝对朝向 |
| `orientation_world` | 保持 `p` | `R' = Q R_p` | 保持 `s_p` | 只按世界/父坐标轴改变朝向 |
| `orientation_local` | 保持 `p` | `R' = R_p Q` | 保持 `s_p` | 只按自身/局部坐标轴改变朝向 |
| `compose_left` | 精确 `G * pose` | 精确左复合 | `s*s_p` | 完整群复合、父系变换 |
| `compose_right` | 精确 `pose * G` | 精确右复合 | `s_p*s` | 机体系偏移、传感器外参 |

其中 `c = about`，默认原点。

### 只改变朝向，不改变位置

```python
turned_world = poses.transform(
    Transform(rotation=[0, 0, 30]),
    mode="orientation_world",
)

turned_local = poses.transform(
    Transform(rotation=[0, 0, 30]),
    mode="orientation_local",
)
```

左乘与右乘在三维旋转中通常不相同。

### 绕点旋转位置，保持每个位姿绝对朝向

```python
orbited = poses.transform(
    Transform(rotation=[0, 0, 90]),
    mode="position",
    about=[10, 0, 0],
)
```

### Sim(3) 对齐一条刚体位姿轨迹

```python
alignment = Transform(
    scale=1.5,
    rotation=[0, 0, 20],
    translation=[2, -1, 0],
)

aligned = poses.transform(alignment, mode="world")
```

`world` 模式会缩放位姿原点之间的距离并左旋每个朝向，但不会把每个刚体局部坐标系本身改成带尺度坐标系。若需要数学上的精确 Sim(3) 群乘法，使用：

```python
exact = poses.transform(alignment, mode="compose_left")
# 等价于 alignment * poses
```

### 局部坐标系中的安装偏移

```python
sensor_offset = Transform(
    translation=[0.2, 0.0, 0.1],
    rotation=[0.0, 0.0, 5.0],
)

sensor_poses = body_poses.transform(sensor_offset, mode="compose_right")
# 等价于 body_poses * sensor_offset
```

### 将 Transform 作为操作符调用

```python
world_points = G(points)
exact_poses = G(poses)  # 默认精确左复合，等价于 G * poses

mapped_poses = G.apply_to_pose(poses, mode="world")
```

## 8. 被动坐标系变换

“主动移动物体”和“更换描述坐标系”数学上相差一个逆变换。若 `G_new_old` 将旧世界坐标映射到新世界坐标：

```python
poses_in_new_world = poses.transform(G_new_old, mode="world")
```

若手中保存的是反方向 `G_old_new`：

```python
poses_in_new_world = poses.transform(~G_old_new, mode="world")
```

涉及严格坐标系链时，优先使用带 `wrd_frame` / `ego_frame` 标签的 `*`、`compose_left` 或 `compose_right`，避免只凭名称猜测方向。

## 9. 索引、变形和拼接

```python
first = poses[0]             # Pose
subset = poses[:2]           # PoseArray
flat = poses.ravel()
grid = flat.reshape(2, -1)
combined = PoseArray.concatenate([subset, subset])
```

也可以从单个位姿打包：

```python
packed = PoseArray.from_poses([pose_a, pose_b, pose_c])
```
