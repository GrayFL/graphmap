import subprocess
import sys


def test_public_imports_from_outside_repository(tmp_path):
    subprocess.run([sys.executable, "-c", """
import sys
import importlib.abc
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {
            'cv2', 'yaml', 'munch', 'scipykit', 'dataset_wrapper',
            'rsim', 'mujoco', 'mujocosim', 'pyrobo', 'matplotlib',
        }:
            raise ImportError('optional dependency blocked: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from graphmap.pose import Pose, PoseArray
from graphmap.sensor import CameraIntrinsic
from graphmap import Environment
from graphmap.traversability import GroundConfig, GroundMap
from graphmap.adapters import pointcloud_observation
env = Environment()
env.observe(**pointcloud_observation([[1, 0, 0]], pose_sensor=Pose()))
assert env.N == 1
assert 'cv2' not in sys.modules
assert 'graphmap.map2d' not in sys.modules
navigation = GroundMap(GroundConfig())
navigation.update([[0., 0., 0.]], robot_pose=Pose(), ground_z=0)
assert 'cv2' not in sys.modules
"""], cwd=tmp_path, check=True, timeout=60)


def test_root_exports_resolve():
    import pytest
    pytest.importorskip("cv2")
    import graphmap
    for name in graphmap.__all__:
        assert getattr(graphmap, name) is not None
