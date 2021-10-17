from typing import List, Tuple
import os

import cv2
import numpy as np
import torch


def convert_np_for_vis(
    img: np.array,
    target_size: Tuple[int] = [256, 256],
) -> np.array:
    """Convert np.array to open-cv format.

    Args:
        img (np.array): np.array to be converted
        target_size (Tuple[int], optional): Size of returned image.
                                            Defaults to [256, 256].

    Returns:
        [np.array]: Converted np.array
    """
    img = transpose_to_channels_first(img)
    img = cv2.resize(img, target_size)

    img = np.clip(img, -2, 2)
    img = (img + 2) * 255 / 4
    img = img.astype(np.uint8)
    return img


def transpose_to_channels_first(np_arrray: np.array) -> np.array:
    """Transpose np.array to open-cv format"""
    if np_arrray.ndim == 3:
        np_arrray = np.transpose(np_arrray, [1, 2, 0])
    return np_arrray


def np_to_torch(img_np: np.array, dtype=torch.float) -> torch.Tensor:
    """Convert np.array to torch.Tensor."""
    if img_np.dtype != np.float32:
        img_np = img_np.astype(np.float32)
    img_tensor = torch.from_numpy(img_np)

    img_tensor = img_tensor.type(dtype)

    return img_tensor


def is_npy_cropped(path: str, crop_size: List[int]):

    img = np.load(path)
    height, width = (img.shape[0], img.shape[1])
    if height < crop_size[0] or width < crop_size[1]:
        raise ValueError(
            "Raster cannot have smaller size than crop size. "
            + f"Raster's size is [{height}, {width}], crop size: {crop_size}"
        )
    if height > crop_size[0] or width > crop_size[1]:
        return True
    else:
        return False


def crop_npy(path: str, dest_dir: str, crop_size: List[int]):
    """Crop np array into subgrids.
       Cropped subgrids cover whole array.
    Args:
        input_img (str): Path to raster file
        dest_dir (str): Destination directory. Must not exist.
        crop_size (List[int]): dimensions of the subgrid
    """
    files = []

    img = np.load(path)
    height, width = (img.shape[0], img.shape[1])

    lat_crop_num = height // crop_size[0]
    long_crop_num = width // crop_size[1]

    for lat_idx in range(lat_crop_num + 1):
        for long_idx in range(long_crop_num + 1):

            x_max = min(height, (lat_idx + 1) * crop_size[0])
            y_max = min(width, (long_idx + 1) * crop_size[1])

            x_min = x_max - crop_size[0]
            y_min = y_max - crop_size[1]

            img_cropped = img[x_min:x_max, y_min:y_max, :]
            raster_name = os.path.splitext(os.path.split(path)[1])[0]
            out_path = os.path.join(dest_dir, f"{raster_name}_{x_min}_{y_min}_{x_max}_{y_max}.npy")

            np.save(out_path, img_cropped)
            files.append(out_path)
