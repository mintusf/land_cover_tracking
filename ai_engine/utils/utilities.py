from copy import deepcopy
import os

import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from ai_engine.utils.io_utils import load_yaml


def split_sample_name(sample_name: str) -> str:
    """Split sample name into ROI folder name, area, and subgrid ID."""
    parts = sample_name.split("_")
    roi_folder_name = "_".join(parts[:2])
    area = parts[2]
    subgrid_id = parts[3]
    return roi_folder_name, area, subgrid_id


def get_gpu_count(cfg: CfgNode) -> int:
    """Returns used GPUs count given config and mode"""

    device = cfg.INFER.DEVICE
    if "cpu" in device:
        devices = 1
    elif "all" in device:
        devices = torch.cuda.device_count()
    else:
        devices = len(device.split(":")[1].split(","))
    return devices
