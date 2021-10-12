import cv2
import os
from typing import Tuple, Union
import numpy as np
import json

# TODO: remove these hardcodings after adding submodule with model
channels_stats = {
    "means": {
        "B1": 1465.7796630859375,
        "B2": 1229.3665771484375,
        "B3": 1138.5279541015625,
        "B4": 1135.537353515625,
        "B5": 1347.8526611328125,
        "B6": 1939.2718505859375,
        "B7": 2221.37255859375,
        "B8": 2163.6171875,
        "B8a": 2420.10986328125,
        "B9": 786.05615234375,
        "B10": 23.309160232543945,
        "B11": 1994.876220703125,
        "B12": 1343.784912109375,
    },
    "stds": {
        "B1": 756.1244670490572,
        "B2": 750.332666316304,
        "B3": 748.0020722062175,
        "B4": 963.1195533211783,
        "B5": 947.8419869546132,
        "B6": 986.349333394727,
        "B7": 1083.805370731563,
        "B8": 1057.8116326811482,
        "B8a": 1136.0447622169102,
        "B9": 579.574680618541,
        "B10": 31.745665139433658,
        "B11": 1117.3512428732556,
        "B12": 975.1746865197342,
    },
}
channels = [
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


def convert_sat_np_for_vis(
    img_path: str,
    target_size: Union[None, Tuple[int]] = None,
) -> np.array:
    """Convert np.array to open-cv format.
    Args:
        img_path (str): np.array to be converted
        target_size (Tuple[int], optional): Size of returned image.
                                            Defaults to [256, 256].
    Returns:
        [np.array]: Converted np.array
    """
    img = np.load(img_path)
    img = img.astype(np.float32)
    if target_size is not None:
        img = cv2.resize(img, target_size)

    for channel in [0, 1, 2, 3]:
        img[:, :, channel] = (
            img[:, :, channel] - channels_stats["means"][channels[channel]]
        ) / channels_stats["stds"][channels[channel]]

    img = img[:, :, [3, 2, 1]]

    img = np.clip(img, -2, 2)
    img = (img + 2) * 255 / 4
    img = img.astype(np.uint8)
    return img


import glob


def get_next_folder_name(dir: str) -> str:
    """Get a string of integer of the next folder in directory"""
    files = glob.glob(dir + "/*/")
    if len(files) == 0:
        return "0"
    else:
        all_folders_ids = [int(os.path.split(os.path.split(f)[0])[-1]) for f in files]
        return str(max(all_folders_ids) + 1)


def load_json(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, new_dict):
    if os.path.isfile(path):
        old_dict = load_json(path)
        old_dict.update(new_dict)
        with open(path, "w") as f:
            json.dump(old_dict, f)
    else:
        with open(path, "w") as f:
            json.dump(new_dict, f)
