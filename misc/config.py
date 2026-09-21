import yaml
from pathlib import Path
from typing import Union, Literal

try:
    from munch import Munch
except ImportError:
    class Munch(dict):
        """Small local fallback for the subset of munch used by graphmap."""

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            self[key] = value

        @classmethod
        def fromDict(cls, value):
            if isinstance(value, dict):
                return cls({
                    key: cls.fromDict(item)
                    for key, item in value.items()
                    })
            if isinstance(value, list):
                return [cls.fromDict(item) for item in value]
            return value


def load_config(config_fp: Union[str, Path]):
    config_fp = Path(config_fp)
    assert config_fp.exists() and config_fp.is_file(), f"Config file {config_fp} not found."
    assert config_fp.suffix in ['.yaml', '.yml'], f"Config file {config_fp} must be a .yaml or .yml file."
    with open(config_fp, 'r') as f:
        config = Munch.fromDict(yaml.safe_load(f))
    if "__base__" in config:
        base_config = load_config(config.__base__)
        config = merge_config(config, base_config)
    return config


def merge_config(config: Munch, base_config: Munch):
    merged_config = Munch.fromDict(base_config)
    for k, v in config.items():
        if (k in merged_config and isinstance(merged_config[k],
                                                (dict, Munch))
                and isinstance(v, (dict, Munch))):
            merged_config[k] = merge_config(v, merged_config[k])
        else:
            merged_config[k] = v
            print(k, v)
    return merged_config
