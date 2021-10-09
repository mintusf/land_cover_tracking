import json
from math import ceil
import os

from sentinelhub import (
    SHConfig,
    bbox_to_dimensions,
    BBox,
    CRS,
    BBoxSplitter,
    SentinelHubRequest,
    DataCollection,
    MimeType,
)
import numpy as np
from shapely.geometry import Polygon


def get_sentinelhub_config(config_file: str):
    config = SHConfig()

    with open(config_file, "r") as f:
        config_vals = json.load(f)

    config.instance_id = config_vals["instance_id"]
    config.sh_client_id = config_vals["sh_client_id"]
    config.sh_client_secret = config_vals["sh_client_secret"]

    return config


def get_total_size(polygon, resolution):
    osm_splitter = BBoxSplitter(
        [polygon], CRS.WGS84, split_shape=[1, 1], reduce_bbox_sizes=True
    )
    cropped_tile = osm_splitter.get_bbox_list()[0]
    size = bbox_to_dimensions(cropped_tile, resolution=resolution)
    return size


def get_tiles_coord(lat, long, resolution, upper_size_margin=2400):
    # rectangle_coord = [lat[0], long[0], lat[1], long[1]]
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


def get_eval_script(bands):
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


def download_raster(tile, savedir, resolution):
    eval_script = get_eval_script(
        [
            "B01",
            "B02",
            "B03",
            "B04",
        ]
    )

    config = get_sentinelhub_config("sentinelhub_config.json")
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
        bbox=tile,
        size=bbox_to_dimensions(tile, resolution=resolution),
        config=config,
    )

    downloaded_img = request_all_bands.get_data()
    return downloaded_img


def get_raster_from_coord(lat, long, savedir):
    # Divide the lat and long into the appropriate number of tiles
    tiles = get_tiles_coord(lat, long, resolution=10)

    # Get all rasters and save them
    os.makedirs(savedir, exist_ok=True)
    coords = {}
    for i, tile in enumerate(tiles):
        img = download_raster(tile, savedir, resolution=10)
        filename = f"tile_{i}"
        np.save(os.path.join(savedir, filename), img[0])
        coords[filename] = [tile.lower_left[::-1], tile.upper_right[::-1]]

    return coords
