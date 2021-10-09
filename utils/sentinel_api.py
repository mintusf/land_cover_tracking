import json
from math import ceil
import os
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Polygon
from sentinelhub import (
    SHConfig,
    bbox_to_dimensions,
    CRS,
    BBox,
    BBoxSplitter,
    SentinelHubRequest,
    DataCollection,
    MimeType,
)


def get_sentinelhub_config(config_file: str) -> SHConfig:
    """Loads sentinel hub config from file and returns it"""
    config = SHConfig()

    with open(config_file, "r") as f:
        config_vals = json.load(f)

    config.instance_id = config_vals["instance_id"]
    config.sh_client_id = config_vals["sh_client_id"]
    config.sh_client_secret = config_vals["sh_client_secret"]

    return config


def get_total_size(polygon: Polygon, resolution: float) -> Tuple[float]:
    """Returns the total size of the polygon in pixels given resolution

    Args:
        polygon (Polygon): The polygon to get the size of
        resolution (float): The resolution of the image (in meters)

    Returns:
        Tuple[float]: Size of the polygon in pixels
    """
    splitter = BBoxSplitter(
        [polygon], CRS.WGS84, split_shape=[1, 1], reduce_bbox_sizes=True
    )
    cropped_tile = splitter.get_bbox_list()[0]
    size = bbox_to_dimensions(cropped_tile, resolution=resolution)
    return size


def get_tiles_coord(
    lat: Tuple[float],
    long: Tuple[float],
    resolution: float,
    upper_size_margin: int = 2400,
) -> List[BBox]:
    """Given lattitude, longitude, resolution and upper limit for a polygon,
       returns a list of coordinates into which the polygon should be divided into

    Args:
        lat (Tuple[float]): The lattitude of the polygon [south, north]
        long (Tuple[float]): The longitude of the polygon [west, east]
        resolution (float): The resolution of the image (in meters)
        upper_size_margin (int, optional): Upper margin for tile size.
                                           Defined by API download limitation.
                                           Defaults to 2400.

    Returns:
        List[BBox]: List of BBoxes which correspond to tiles to which polygon is divided
                    Coordinates in BBox are in format ((west, south),(east, north))
    """
    whole_rectangle = Polygon(
        [[long[0], lat[0]], [long[1], lat[0]], [long[1], lat[1]], [long[0], lat[1]]]
    )
    whole_size = get_total_size(whole_rectangle, resolution)
    osm_splitter = BBoxSplitter(
        [whole_rectangle],
        CRS.WGS84,
        split_shape=[
            ceil(whole_size[0] / upper_size_margin),
            ceil(whole_size[1] / upper_size_margin),
        ],
    )

    cropped_tiles = osm_splitter.get_bbox_list()

    return cropped_tiles


def get_eval_script(bands: List[str]) -> str:
    """Generates API download scipt given list of bands
    according to https://sentinelhub-py.readthedocs.io/en/latest/examples/
                 ogc_request.html?highlight=evalscript#Example-11:-Evalscript"""
    bands_len = len(bands)
    return_string = (",\n                            ").join(
        [f"sample.{band}" for band in bands]
    )
    evalscript = """
                //VERSION=3
                function setup() {
                    return {
                        input: [{
                            bands:"""

    evalscript += f""" [{", ".join([f'"{band}"' for band in bands])}],"""

    evalscript += """
                            units: "DN"
                        }],
                        output: {"""

    evalscript += f"""
                            bands: {bands_len},"""

    evalscript += """
                            sampleType: "INT16"
                        }
                    };
                }

                function evaluatePixel(sample) {
                    return ["""

    evalscript += f"""{return_string}"""

    evalscript += """];
                }
        """

    return evalscript


def download_raster(tile_coord: Tuple[float], resolution: float) -> np.array:
    """Given tile coordinates and resolution, downloads the tile using sentinelhub API

    Args:
        tile_coord (Tuple[float]): The coordinates of the tile
        resolution (float): The resolution of the image (in meters)

    Returns:
        np.array: The downloaded tile with shape (h, w, bands)
    """

    # TODO: get from model config
    eval_script = get_eval_script(
        [
            "B01",
            "B02",
            "B03",
            "B04",
        ]
    )

    config = get_sentinelhub_config("sentinelhub_config.json")
    # TODO: Load dates from config
    request_all_bands = SentinelHubRequest(
        evalscript=eval_script,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=("2020-08-01", "2020-09-30"),
                mosaicking_order="leastCC",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=tile_coord,
        size=bbox_to_dimensions(tile_coord, resolution=resolution),
        config=config,
    )

    downloaded_img = request_all_bands.get_data()[0]
    return downloaded_img


def get_raster_from_coord(
    lat: Tuple[float], long: Tuple[float], resolution: float, savedir: str
) -> Dict[str, Tuple[float]]:
    """Given lattitude, longitude of a polygon and resolution of target tiles,
        divides polygon into tiles, downloads them and save on disk

    Args:
        lat (Tuple[float]): The lattitude of the polygon [south, north]
        long (Tuple[float]): The longitude of the polygon [west, east]
        resolution (float): The resolution of the image (in meters)
        savedir (str): The directory to save the downloaded tiles

    Returns:
        Dict[str, Tuple[float]]: [description]
    """
    # Divide the lat and long into the appropriate number of tiles
    tiles = get_tiles_coord(lat, long, resolution=resolution)

    # Get all rasters and save them
    os.makedirs(savedir, exist_ok=True)
    coords = {}
    for i, tile in enumerate(tiles):
        img = download_raster(tile, resolution=resolution)
        filename = f"tile_{i}"
        np.save(os.path.join(savedir, filename), img)
        tile_bounds = tile.geometry.bounds
        tile_coord = {"lat": [tile_bounds[1], tile_bounds[3]], "long": [tile_bounds[0], tile_bounds[2]]}
        coords[filename] = tile_coord

    return coords
