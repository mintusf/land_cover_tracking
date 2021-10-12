import os

from numpy import random
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from config.default import CfgNode

from ai_engine.dataset.patch_dataset import PatchDataset
from ai_engine.dataset.transforms import get_transform
from ai_engine.utils.utilities import (
    get_gpu_count,
)


def get_dataloader(cfg: CfgNode, samples_list: str) -> DataLoader:
    """Builds and returns a dataloader for the dataset.

    Args:
        cfg (CfgNode): Config object.
        samples_list (str): Either a path to a text file containing the
                                list of samples or one of ["train", "val", "test"].

    Returns:
        DataLoader: [description]
    """

    if not os.path.isfile(cfg.DATASET.INPUT.STATS_FILE):
        raise FileNotFoundError("Channels stats file doesn't exist")

    transform = get_transform(cfg)
    transforms = Compose([transform])

    dataset = PatchDataset(
        cfg, samples_list, transforms=transforms, aug_transforms=None
    )

    num_workers = cfg.INFER.WORKERS
    shuffle = False
    batch_size = cfg.INFER.BATCH_SIZE_PER_DEVICE * get_gpu_count(cfg)
    drop_last = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        worker_init_fn=random.seed(cfg.INFER.SEED),
        drop_last=drop_last,
    )

    return dataloader
