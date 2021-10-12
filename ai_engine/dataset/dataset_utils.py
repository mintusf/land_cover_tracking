from yacs.config import CfgNode

from ai_engine.utils.io_utils import load_yaml


def get_channels_in_count(cfg: CfgNode) -> int:
    """Returns the number of input channels given config"""
    return len(cfg.DATASET.INPUT.USED_CHANNELS)


def get_channels_out_count(cfg: CfgNode) -> int:
    """Returns the number of output channels given config"""
    labels_config = load_yaml(cfg.DATASET.MASK.CONFIG)
    return len(labels_config["class2label"])
