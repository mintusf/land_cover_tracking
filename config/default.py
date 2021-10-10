import os
from yacs.config import CfgNode

_C = CfgNode()

_C.DATA_DIR = "app_data"
_C.POLYGON_JSON_NAME = "polygons.json"
_C.RESOLUTION = 10


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_cfg_from_file(filepath: str) -> CfgNode:
    """Load a config file and return a CfgNode object"""
    cfg = get_cfg_defaults()
    cfg.merge_from_file(filepath)
    cfg.freeze()

    return cfg
