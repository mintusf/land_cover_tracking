import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from yacs.config import CfgNode

from ai_engine.utils.io_utils import get_lines_from_txt, load_yaml
from ai_engine.utils.raster_utils import raster_to_np, np_to_torch


class PatchDataset(Dataset):
    def __init__(
        self, cfg: CfgNode, samples_list: str, transforms=None, aug_transforms=None
    ):
        """Patch Dataset initialization

        Args:
            cfg (CfgNode): Config
            samples_list (str): Either a path to a text file containing the
                                list of samples or one of ["train", "val", "test"].
                                If a path, Dataset is used in inference mode and
                                only input is generated.
            transforms (callable, optional): Optional transform to be applied
            aug_transforms (callable, optional): Optional data augmentation transforms
                                                 to be applied
        """
        self.cfg = cfg

        self.mask_config = load_yaml(cfg.DATASET.MASK.CONFIG)
        self.channels_list = cfg.DATASET.INPUT.CHANNELS
        self.input_used_channels = cfg.DATASET.INPUT.USED_CHANNELS

        assert os.path.isfile(samples_list), f"Invalid samples list path {samples_list}"
        self.dataset_list_path = samples_list
        samples_list = "infer"

        self.mode = samples_list

        self.dataset_list = get_lines_from_txt(self.dataset_list_path, shuffle=True)

        self.transforms = transforms
        self.aug_transforms = aug_transforms

        self.device = cfg.INFER.DEVICE

    def __len__(self) -> int:
        """Get length of dataset

        Returns:
            length (int): Length of dataset
        """
        return len(self.dataset_list)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get single sample given index

        Args:
            index (int): Index

        Returns:
            sample (Dict[str, torch.Tensor]): Sample, including:
                                              * input image
                                              * target mask
        """
        # Get sample name
        sample_name = self.dataset_list[index]

        # Get input numpy array
        input_raster_path = sample_name

        ext = os.path.splitext(input_raster_path)[1]
        if ext == ".tiff":
            input_np = raster_to_np(input_raster_path, bands=self.input_used_channels)
        elif ext == ".npy":
            input_np = np.load(input_raster_path)
            input_np = input_np[:, :, self.input_used_channels]
            input_np = np.transpose(input_np, [2, 0, 1])
        else:
            raise NotImplementedError(
                f"Extension {ext} is not supported as model's input"
            )

        if "cuda" in self.device:
            if "all" in self.device:
                device = 0
            else:
                devices = self.device.split(":")[1].split(",")
                device = devices[0]
            device = torch.device(f"cuda:{device}")
        elif "cpu" in self.device:
            device = torch.device("cpu")
        else:
            raise NotImplementedError

        input_tensor = np_to_torch(input_np)
        input_tensor = input_tensor.to(device).float()

        sample = {"input": input_tensor, "name": sample_name}

        # Transform
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
