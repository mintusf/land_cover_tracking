import glob
import os
from typing import Dict, Tuple, List

import cv2

from utils.ai_engine_wrapper import ai_engine_infer
from utils.sentinel_api import get_raster_from_coord
from utils.io_utils import (
    convert_sat_np_for_vis,
    get_next_folder_name,
    write_json,
    load_json,
)
from config.default import CfgNode


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

        # TODO: refactor to utils method
        tile_coord = coords[png_path]
        converted_coord = [
            [tile_coord["lat"][0], tile_coord["long"][0]],
            [tile_coord["lat"][1], tile_coord["long"][1]],
        ]
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
    whole_img = cv2.imread(
        os.path.join(config.DATA_DIR, str(polygon_id), f"{tile_name}.png")
    )
    for pred_path in glob.glob(
        os.path.join(
            config.DATA_DIR, str(polygon_id), f"{tile_name}", "alphablend", "*.png"
        )
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
    for input_file in glob.glob(
        os.path.join(config.DATA_DIR, selected_polygon_pred, "*.npy")
    ):
        tile_name = os.path.splitext(os.path.split(input_file)[1])[0]
        ai_engine_infer(
            config,
            tile_path=input_file,
            checkpoint=config.INFER.WEIGHTS_PATH,
            destination=os.path.join(config.DATA_DIR, selected_polygon_pred, tile_name),
        )

        savedir = os.path.join(
            config.DATA_DIR, selected_polygon_pred, f"{tile_name}_pred.png"
        )
        merge_preds(selected_polygon_pred, tile_name, savedir, config)
        coords = load_json(os.path.join(config.DATA_DIR, config.POLYGON_JSON_NAME))
        tile_coord = coords[input_file.replace("npy", "png")]
        converted_coord = [
            [tile_coord["lat"][0], tile_coord["long"][0]],
            [tile_coord["lat"][1], tile_coord["long"][1]],
        ]
        paths.append(savedir)
        coords_collected.append(converted_coord)

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
        converted_coord = [
            [tile_coord["lat"][0], tile_coord["long"][0]],
            [tile_coord["lat"][1], tile_coord["long"][1]],
        ]

        paths.append(url)
        coords.append(converted_coord)

    return paths, coords
