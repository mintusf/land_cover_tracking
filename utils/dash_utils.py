import glob
import os
from typing import Dict, Tuple, List

import cv2
import numpy as np
import torch

from utils.ai_engine_wrapper import ai_engine_infer
from utils.sentinel_api import get_raster_from_coord
from utils.io_utils import (
    convert_sat_np_for_vis,
    get_next_folder_name,
    write_json,
    load_json,
    load_yaml,
)
from ai_engine.utils.infer_utils import get_path_for_output
from config.default import CfgNode
from ai_engine.utils.visualization_utils import generate_save_alphablend


def get_coord_from_feature(feature):
    bounds = feature["properties"]["_bounds"]
    return f"lat {bounds[0]['lat']:.2f}, long {bounds[0]['lng']:.2f}"


def get_polygon_coord(polygons, selected_polygon):
    return [
        [
            polygons["features"][selected_polygon]["properties"]["_bounds"][0]["lat"],
            polygons["features"][selected_polygon]["properties"]["_bounds"][1]["lat"],
        ],
        [
            polygons["features"][selected_polygon]["properties"]["_bounds"][0]["lng"],
            polygons["features"][selected_polygon]["properties"]["_bounds"][1]["lng"],
        ],
    ]


def get_coord_single_key(
    coord: Dict[str, List[float]], vertical: str, horizontal: str
) -> List[float]:
    """Returns coordinates of a selected corner

    Args:
        coord (Dict[str, List[float]]): Polygon coordinates
        vertical (str): String from ["top", "bottom"] indicating the corner
        horizontal (str): String from ["left", "right"] indicating the corner

    Returns:
        List[float]: [latitude, longitude] coordinates of the corner
    """
    idx_vertical = 1 if vertical == "top" else 0
    idx_horizontal = 0 if horizontal == "top" else 1
    return [coord["lat"][idx_vertical], coord["long"][idx_horizontal]]


def get_coord_multiple_keys(
    keys: List[str],
    coords: Dict[str, Dict[str, List[float]]],
    vertical: int,
    horizontal: int,
) -> Tuple[List[float]]:
    """Returns lists of corner coordinates of all polygons

    Args:
        keys (List[str]): List containing paths to polygon files
        coords (Dict[str, Dict[str, List[float]]]): [description]
        vertical (str): String from ["top", "bottom"] indicating the corner
        horizontal (str): String from ["left", "right"] indicating the corner

    Returns:
        Tuple[List[float]]: Lists of corner coordinates of all polygons
    """
    lat_coords = []
    long_coords = []

    for key in keys:
        coord = coords[key]
        lat_coord, long_coord = get_coord_single_key(coord, vertical, horizontal)
        lat_coords.append(lat_coord)
        long_coords.append(long_coord)

    return lat_coords, long_coords


def get_corner_coord(
    polygon_id: str, vertical: str, horizontal: str, config: CfgNode
) -> List[float]:
    """Returns [lattitude, longitude] coordinates of a polygon corner

    Args:
        polygon_id (str): Id of the selected polygon
        vertical (str): String from ["top", "bottom"] indicating the corner
        horizontal (str): String from ["left", "right"] indicating the corner
        config (CfgNode): App config

    Returns:
        List[float]: Corner's coordinates
    """
    coords = load_json(os.path.join(config.DATA_DIR, config.POLYGON_JSON_NAME))
    directory = os.path.join(config.DATA_DIR, str(polygon_id))
    keys = glob.glob(os.path.join(directory, "tile_*.png"))
    keys = [key for key in keys if "pred" not in key]
    assert vertical in ["top", "bottom"]
    comp_func_vertical = max if vertical == "top" else min

    assert horizontal in ["left", "right"]
    comp_func_horizontal = max if horizontal == "right" else min

    vertical_coords, horizontal_coords = get_coord_multiple_keys(
        keys, coords, vertical, horizontal
    )

    vertical_coord = comp_func_vertical(vertical_coords)
    horizontal_coord = comp_func_horizontal(horizontal_coords)

    return [vertical_coord, horizontal_coord]


def download_action(
    polygons: dict, selected_polygon_download: str, config: CfgNode
) -> Tuple[List[str], List[float]]:
    """Downloads satellite data using sentinel API.
        Due to API limitation, if a selected polygon is too big,
        it is splited into smaller ones which are downloaded instead.
        Returns paths to downloaded images and corresponding coordinates.

    Args:
        polygons (dict): Dictioneries with coordinates of all polygons
        selected_polygon_download (str): Id of selected polygon for download
        config (CfgNode): App config

    Returns:
        Tuple[List[str], List[float]]: A Tuple containing
            * List of paths to downloaded image
            * List of corresponding coordinates, in format:
                [[south, west], [north, east]]
    """
    coord = get_polygon_coord(polygons, int(selected_polygon_download))

    foldername = get_next_folder_name(config.DATA_DIR)
    savedir = os.path.join(config.DATA_DIR, foldername)
    coords = get_raster_from_coord(coord[0], coord[1], config, savedir)

    write_json(os.path.join(config.DATA_DIR, config.POLYGON_JSON_NAME), coords)

    img_paths = glob.glob(f"{savedir}/*.npy")
    paths = []
    coords_collected = []

    for img_path in img_paths:
        img = convert_sat_np_for_vis(img_path)
        png_path = img_path.replace(".npy", ".png")
        cv2.imwrite(png_path, img)

        converted_coord = get_converted_coords_from_dict(coords, png_path)

        paths.append(png_path)
        coords_collected.append(converted_coord)

    return paths, coords_collected


def merge_preds(polygon_id: str, tile_name: str, savedir: str, config: CfgNode) -> None:
    """Merges subgrid predictions to build an image for whole polygon

    Args:
        polygon_id (str): Id of polygon selected for prediction
        tile_name (str): Name of the tile from the polygon
        savedir (str): Saving directory
        config (CfgNode): App config
    """
    polygon_dir = os.path.join(config.DATA_DIR, str(polygon_id))
    whole_img = cv2.imread(os.path.join(polygon_dir, f"{tile_name}.png"))
    for pred_path in glob.glob(
        os.path.join(polygon_dir, f"{tile_name}", "alphablend", "*.png")
    ):
        parts = os.path.splitext(os.path.split(pred_path)[1])[0].split("_")
        x_min = int(parts[2])
        y_min = int(parts[3])
        x_max = int(parts[4])
        y_max = int(parts[5])
        subgrid = cv2.imread(pred_path)
        whole_img[x_min:x_max, y_min:y_max, :] = subgrid

    cv2.imwrite(savedir, whole_img)


def predict_action(
    config: CfgNode, selected_polygon_pred: str
) -> Tuple[List[str], List[float]]:
    """Performs prediction on selected downloaded are

    Args:
        config (CfgNode): App config
        selected_polygon_pred (str): Id of selected area, corresponds to folder name

    Returns:
        Tuple[List[str], List[float]]: A Tuple containing
            * List of paths to downloaded image
            * List of corresponding coordinates, in format:
                [[south, west], [north, east]]
    """

    paths = []
    coords_collected = []
    input_files_dir = os.path.join(config.DATA_DIR, selected_polygon_pred)
    for input_file in glob.glob(os.path.join(input_files_dir, "*.npy")):
        tile_name = os.path.splitext(os.path.split(input_file)[1])[0]
        ai_engine_infer(
            config,
            tile_path=input_file,
            checkpoint=config.INFER.WEIGHTS_PATH,
            destination=os.path.join(input_files_dir, tile_name),
        )

        savedir = os.path.join(input_files_dir, f"{tile_name}_pred.png")
        merge_preds(selected_polygon_pred, tile_name, savedir, config)

        paths.append(savedir)

        converted_coords = get_converted_coords(config, input_file)
        coords_collected.append(converted_coords)

    return paths, coords_collected


def generate_alpha_for_tile(mask_file: str, mask_config: dict, alpha: float) -> None:
    """Generates alphablend for a single tile

    Args:
        mask_file (str): Path to the predicted tile mask
        mask_config (dict): Mask config, should have
                            "alpha", "class2label" and "colors" defined
        alpha (float): Alpha for alphablend
    """
    input_single_file = mask_file.replace("/mask_np", "")
    mask = np.load(mask_file)
    input_img = convert_sat_np_for_vis(input_single_file)

    name = os.path.splitext(os.path.split(mask_file)[1])[0]
    destination = os.path.split(os.path.split(mask_file)[0])[0]
    output_path = get_path_for_output("alphablend", destination, name)

    generate_save_alphablend(
        input_img,
        mask,
        mask_config,
        output_path,
        alpha,
    )


def convert_coords(coords_in: Dict[str, List[float]]) -> List[List[float]]:
    """Converts coordinates from a dict to a list

    Args:
        coords_in (Dict[str, List[float]]): Dictionary with following elements:
            'lat':
                [south, north]
            'long':
                [west, east]

    Returns:
        List[List[float]]: Coordinates in format [[south, west], [north, east]]
    """
    converted_coord = [
        [coords_in["lat"][0], coords_in["long"][0]],
        [coords_in["lat"][1], coords_in["long"][1]],
    ]
    return converted_coord


def get_converted_coords_from_dict(
    coords_dict: Dict[str, Dict[str, List[float]]], key: str
) -> List[List[float]]:
    """Returns covnerted coordinates from a dictionary given a key

    Args:
        coords_dict (Dict[Dict[str, float]]): Dictionary with following elements:
            path_to_a_png_tile_file:
                'lat':
                    [south value, north value]
                'long':
                    [west value, east value]

        key (str): Key in a dictionary

    Returns:
        List[List[float]]: Coordinates in format [[south, west], [north, east]]
    """
    tile_coord = coords_dict[key]

    converted_coords = convert_coords(tile_coord)
    return converted_coords


def get_converted_coords(config: CfgNode, input_file: str) -> List[List[float]]:
    """Returns coordinates given a path to an image.
       Supported image extensions are: [npy, png]

    Args:
        config (CfgNode): App config
        input_file (str): Path to an image

    Returns:
        List[List[float]]: Coordinates in format [[south, west], [north, east]]
    """
    coords = load_json(os.path.join(config.DATA_DIR, config.POLYGON_JSON_NAME))
    png_path = input_file.replace("npy", "png")
    converted_coord = get_converted_coords_from_dict(coords, png_path)
    return converted_coord


def new_alpha_action(
    config: CfgNode, selected_polygon_analyze: str, alpha
) -> Tuple[List[str], List[float]]:
    """Performs prediction on selected downloaded are

    Args:
        config (CfgNode): App config
        selected_polygon_pred (str): Id of selected area, corresponds to folder name

    Returns:
        Tuple[List[str], List[float]]: A Tuple containing
            * List of paths to downloaded image
            * List of corresponding coordinates, in format:
                [[south, west], [north, east]]
    """

    paths = []
    coords_collected = []
    mask_config = load_yaml(config.DATASET.MASK.CONFIG)
    polygon_root = os.path.join(config.DATA_DIR, selected_polygon_analyze)
    for input_file in glob.glob(os.path.join(polygon_root, "*.npy")):
        tile_name = os.path.splitext(os.path.split(input_file)[1])[0]
        save_filename = f"{tile_name}_pred_" + f"{alpha:.02f}".replace(".", "") + ".png"
        savedir = os.path.join(polygon_root, save_filename)

        if not os.path.isfile(savedir):
            tile_masks_dir = os.path.join(polygon_root, tile_name, "mask_np")
            tile_masks = glob.glob(os.path.join(tile_masks_dir, "*.npy"))
            for mask_file in tile_masks:
                generate_alpha_for_tile(mask_file, mask_config, alpha)

            merge_preds(selected_polygon_analyze, tile_name, savedir, config)

        paths.append(savedir)

        converted_coords = get_converted_coords(config, input_file)
        coords_collected.append(converted_coords)

    return paths, coords_collected


def refresh_action(config: CfgNode) -> Tuple[List[str], List[float]]:
    """Collects all available images, both raw and predictions,
        end returns them together with corresponding coordinates.

    Args:
        config (CfgNode): App config

    Returns:
        Tuple[List[str], List[float]]: A Tuple containing
            * List of paths to downloaded image
            * List of corresponding coordinates, in format:
                [[south, west], [north, east]]
    """
    coords_all = load_json(os.path.join(config.DATA_DIR, config.POLYGON_JSON_NAME))
    paths = []
    coords = []
    for key, tile_coord in coords_all.items():
        pred_path = key.replace(".png", "_pred.png")
        if os.path.isfile(pred_path):
            url = pred_path
        else:
            url = key
        converted_coords = convert_coords(tile_coord)

        paths.append(url)
        coords.append(converted_coords)

    return paths, coords


def get_classes_count(polygon_id: str, config: CfgNode) -> Dict[str, int]:
    """Calculates count of each class for the polygon

    Args:
        polygon_id (str): Polygon id
        config (CfgNode): App config

    Returns:
        Dict[str, int]: Classes counts
    """
    polygon_dir = os.path.join(config.DATA_DIR, str(polygon_id))
    all_masks = glob.glob(os.path.join(polygon_dir, "*", "mask_np", "*.npy"))
    mask_config = load_yaml(config.DATASET.MASK.CONFIG)
    class2label = mask_config["class2label"]
    all_counts = np.zeros(len(class2label))
    for mask_path in all_masks:
        mask = np.load(mask_path)
        classes_count = np.bincount(mask.flatten(), minlength=len(class2label))
        all_counts += classes_count

    labels_count = {}
    for class_id, label in class2label.items():
        labels_count[label] = all_counts[class_id]

    return labels_count


def get_top_labels(labels_counts: Dict[str, int], k: int) -> Tuple[np.array, List[str]]:
    """Returns top k classes with highest pixels count

    Args:
        labels_counts (Dict[str, int]): Input dictionary with classes counts
        k (int): Top k to select

    Returns:
        Tuple[np.array, List[str]]: Dictionary with topk classes
    """
    sorted_labels = sorted(labels_counts.keys(), key=lambda x: labels_counts[x])[::-1]
    counts = []
    labels = []
    for i, label in enumerate(sorted_labels):
        if i == k:
            break
        if labels_counts[label]:
            counts.append(labels_counts[label])
            labels.append(label)
    return np.array(counts), labels
