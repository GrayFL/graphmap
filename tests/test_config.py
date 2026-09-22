import pytest

from graphmap.misc.config import load_config


def test_yaml_base_supports_legacy_cwd_and_sibling_paths(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text("map:\n  resolution: 0.05\n  side_length: 10\n")
    (configs / "legacy.yaml").write_text("__base__: configs/base.yaml\nmap:\n  resolution: 0.1\n")
    (configs / "sibling.yaml").write_text("__base__: base.yaml\nmap:\n  resolution: 0.2\n")
    monkeypatch.chdir(tmp_path)
    legacy = load_config(configs / "legacy.yaml")
    sibling = load_config(configs / "sibling.yaml")
    assert legacy.map.resolution == .1
    assert sibling.map.resolution == .2
    assert legacy.map.side_length == sibling.map.side_length == 10
