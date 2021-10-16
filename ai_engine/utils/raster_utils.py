from typing import List, Tuple, Union
import os

import cv2
import numpy as np
import rasterio as rio
from rasterio import mask
from shapely.geometry import Polygon
import torch


def raster_to_np(
    raster_path: str, bands: Tuple[int] = None, dtype=np.float32
) -> np.array:
    """Convert img raster to numpy array. Raster can have any number of bands.
    Args:
        raster_path (str): Path to WV .img file
        bands (Tuple[int]): Tuple of bands to extract
    Returns:
        np.array: raster converted into np.array
    """
    with rio.open(raster_path) as src:
        if bands is None:
            bands = [src.read(band_idx + 1) for band_idx in range(src.count)]
        else:
            bands = [src.read(band_idx + 1) for band_idx in bands]

    img_np = np.array(bands, dtype=dtype)

    return img_np


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


def np_to_raster(img_np: np.array, ref_img: str, savepath: str):
    """Convert np.array to raster and save
    Args:
        img_np (np.array): Image to be saved
        ref_img (str): Referenced raster
        savepath (str): Output raster savepath (tif format is recommended)
    """
    with rio.open(ref_img) as src:
        transform = src.transform
        size = (src.height, src.width)

    with rio.open(
        savepath,
        "w",
        driver="GTiff",
        dtype=img_np.dtype,
        height=size[0],
        width=size[1],
        count=3,
        crs=src.crs,
        transform=transform,
    ) as dst:
        dst.write(img_np)


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
    """Crop raster into subgrids
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

    for lat_idx in range(lat_crop_num):
        for long_idx in range(long_crop_num):

            x_min = lat_idx * crop_size[0]
            y_min = long_idx * crop_size[1]

            x_max = (lat_idx + 1) * crop_size[0]
            y_max = (long_idx + 1) * crop_size[1]

            img_cropped = img[x_min:x_max, y_min:y_max,:]
            raster_name = os.path.splitext(os.path.split(path)[1])[0]
            out_path = os.path.join(dest_dir, f"{raster_name}_{lat_idx}_{long_idx}.npy")

            np.save(out_path, img_cropped)
            files.append(out_path)
