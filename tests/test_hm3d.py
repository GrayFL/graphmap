import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from graphmap import Environment
from graphmap.adapters import rgbd_observation
from graphmap.adapters.hm3d import load_hm3d_frame
from graphmap.pose import Pose
from graphmap.sensor import CameraIntrinsic


def test_export_reader_channel_order_and_optional_semantics(tmp_path):
    cv2 = pytest.importorskip("cv2")
    cv2.imwrite(str(tmp_path / "000003_rgb.png"), np.full((2, 4, 3), [1, 2, 3], np.uint8))
    cv2.imwrite(str(tmp_path / "000003_depth.png"), np.full((2, 4), 1500, np.uint16))
    frame = load_hm3d_frame(tmp_path, 3)
    assert_array_equal(frame["rgb"][0, 0], [3, 2, 1])
    assert frame["semantic"] is None
    np.savez(tmp_path / "000003_semantic.npz", semantic=np.full((2, 4), 42, np.uint32))
    assert load_hm3d_frame(tmp_path, 3)["semantic"][0, 0] == 42
    with pytest.raises(FileNotFoundError):
        load_hm3d_frame(tmp_path, 4)


@pytest.mark.dataset
def test_local_hm3d_rgbd_semantics_roundtrip(tmp_path):
    root = os.environ.get("GRAPHMAP_HM3D_ROOT")
    if not root:
        pytest.skip("set GRAPHMAP_HM3D_ROOT to enable the local dataset integration")
    pytest.importorskip("cv2")
    scene = Path(root) / os.environ.get("GRAPHMAP_HM3D_SCENE", "0000")
    frame = load_hm3d_frame(scene, 0)
    assert frame["semantic"] is not None, "select a scene/frame with semantic labels"
    height, width = frame["depth"].shape
    # This test checks data alignment/roundtrip. Scene calibration/trajectory
    # are covered by the notebook; no scene pose is inferred here.
    camera = CameraIntrinsic.from_sensor(width=width, height=height, fx=width/2, fy=width/2)
    obs = rgbd_observation(frame["depth"], intrinsic=camera, pose_camera=Pose(),
                           rgb=frame["rgb"], attributes={"semantic_id": frame["semantic"]},
                           depth_scale=.001, stride=32, timestamp=0.)
    assert len(obs["points_world"]) > 0
    h, w = obs["pixel_indices"].T
    labels = frame["semantic"][h, w]
    assert obs["extra_keys"]["semantic_id"].get_batch(range(len(h))) == labels.tolist()
    env = Environment()
    env.observe(**obs)
    assert env.index_db("semantic_id").num_links == env.N
    assert all(value in labels for value in env.get_index_values("semantic_id"))
    env.save(tmp_path)
    restored = Environment.load(tmp_path)
    assert restored.get_index_values("semantic_id") == env.get_index_values("semantic_id")
