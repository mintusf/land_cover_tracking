from copy import deepcopy
import json
from typing import List
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from ai_engine.utils.raster_utils import get_stats
from ai_engine.utils.io_utils import get_lines_from_txt, load_yaml


def split_sample_name(sample_name: str) -> str:
    """Split sample name into ROI folder name, area, and subgrid ID."""
    parts = sample_name.split("_")
    roi_folder_name = "_".join(parts[:2])
    area = parts[2]
    subgrid_id = parts[3]
    return roi_folder_name, area, subgrid_id


def get_area_foldername(sensor: str, area: str) -> str:
    """Get area foldername given sensor and area"""
    return f"{sensor}_{area}"


def get_raster_filepath(rootdir: str, sample_name: str, sensor: str) -> str:
    """Get raster filepath given rootdir, sample name, and sensor
    Args:
        rootdir (str): root directory of the dataset
        sample_name (str): sample name, e.g "ROIs2017_winter_27_p36"
        sensor (str): sensor name

    Returns:
        str: raster filepath
    """
    roi_folder_name, area, subgrid_id = split_sample_name(sample_name)
    folder = os.path.join(rootdir, roi_folder_name, get_area_foldername(sensor, area))
    filename = f"{roi_folder_name}_{sensor}_{area}_{subgrid_id}.tif"
    return os.path.join(folder, filename)


def get_sample_name(filename: str) -> str:
    """Get sample name from filename."""
    split = filename.split("_")
    return "_".join(split[:2] + split[3:])


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


def get_single_dataloader(dataloader, cfg, idx, out_loaders_count):
    """Split a dataloader into two dataloaders"""
    single_loader_samples = len(dataloader.dataset) // out_loaders_count

    subgrids_dataset = deepcopy(dataloader.dataset)
    subgrids_dataset.dataset_list = dataloader.dataset.dataset_list[
        idx * single_loader_samples : (idx + 1) * single_loader_samples
    ]

    dataloader_single = DataLoader(
        subgrids_dataset,
        batch_size=cfg.INFER.BATCH_SIZE_PER_DEVICE * get_gpu_count(cfg),
        num_workers=cfg.INFER.WORKERS,
        shuffle=False,
        drop_last=True,
    )

    return dataloader_single


def is_intersection_empty(dataloader1: DataLoader, dataloader2: DataLoader) -> bool:
    """Checks if no sample in both train and checked dataloader"""
    samples1 = set(dataloader1.dataset.dataset_list)
    samples2 = set(dataloader2.dataset.dataset_list)
    return samples1.isdisjoint(samples2)


def get_class_labels_ordered(cfg: CfgNode) -> int:
    """Returns the labels of classes"""
    labels_config = load_yaml(cfg.DATASET.MASK.CONFIG)
    class2label = labels_config["class2label"]
    labels = [class2label[i] for i in range(len(class2label))]
    return labels
