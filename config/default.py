import os
from yacs.config import CfgNode

_C = CfgNode()

_C.DATA_DIR = "app_data"
_C.POLYGON_JSON_NAME = "polygons.json"
_C.RESOLUTION = 10

_C.MODEL = CfgNode()
_C.MODEL.TYPE = "hrnet"
_C.MODEL.CONFIG = "ai_engine/config/model/hrnet.yml"

_C.DATASET = CfgNode()
_C.DATASET.INPUT = CfgNode()
_C.DATASET.INPUT.CHANNELS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8a",
    "B9",
    "B10",
    "B11",
    "B12",
]
_C.DATASET.INPUT.USED_CHANNELS = [1, 2, 3]
_C.DATASET.INPUT.STATS_FILE = os.path.join(
    "ai_engine",
    "config",
    "dataset",
    "stats",
    "channels_stats_less_classes_more_snow.json",
)
_C.DATASET.MASK = CfgNode()
_C.DATASET.MASK.CONFIG = os.path.join(
    "ai_engine", "config", "dataset", "mask_configs", "less_classes.yml"
)
_C.DATASET.SHAPE = [256, 256]

_C.INFER = CfgNode()
_C.INFER.DEVICE = "cpu"
_C.INFER.WORKERS = 0
_C.INFER.BATCH_SIZE_PER_DEVICE = 1
_C.INFER.SEED = 42
_C.INFER.WEIGHTS_PATH = (
    "weights/cfg_weighted_loss_more_snow_data_aug_hrnet_3bands_resume_best_f1.pth"
)


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
