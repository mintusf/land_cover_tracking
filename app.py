from dash import Dash, callback_context
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_leaflet as dl
from flask import Flask, send_from_directory
import glob

import os
import cv2
from shutil import rmtree

from config.default import get_cfg_from_file
from utils.dash_utils import get_coord_from_feature
from utils.sentinel_api import get_raster_from_coord
from utils.io_utils import (
    convert_sat_np_for_vis,
    get_next_folder_name,
    write_json,
    load_json,
)

server = Flask(__name__)
app = Dash(server=server)
config = get_cfg_from_file("config/default.yml")

app.layout = html.Div(
    [
        html.H1(
            id="title",
            children="Land Cover Tracking",
            style={"textAlign": "center", "fontSize": 60},
        ),
        # Setup a map with the edit control.
        dl.Map(
            center=[56, 10],
            zoom=4,
            bounds=[[50, 20], [50.5, 20.5]],
            children=[
                dl.TileLayer(),
                dl.Marker(
                    position=[0, 0],
                    id="marker",
                ),
                dl.FeatureGroup([dl.EditControl(id="edit_control")]),
            ],
            style={
                "width": "100%",
                "height": "80vh",
                "margin": "auto",
                "display": "block",
                "position": "relative",
            },
            id="map",
        ),
        html.Button(
            id="download_raster",
            children=html.H2(
                "Download raster", style={"textAlign": "center", "fontSize": 30}
            ),
            style={
                "display": "inline-block",
                "textalign": "center",
            },
        ),
        dcc.Dropdown(
            id="polygon-dropdown",
            placeholder="Please choose the polygon",
            value="None",
            style={
                "width": "80%",
                "height": "4vh",
                "fontSize": 30,
                "display": "inline-block",
            },
        ),
    ]
)

DATA_DIR = config.DATA_DIR
POLYGON_JSON_NAME = config.POLYGON_JSON_NAME
RESOLUTION = config.RESOLUTION
rmtree(DATA_DIR, ignore_errors=True)


@server.route(f"/{DATA_DIR}/<path:path>")
def get_mask_url(path):
    """Serve a file from the upload directory."""
    img = send_from_directory(DATA_DIR, path, as_attachment=True, cache_timeout=1)
    return img


@app.callback(
    Output("polygon-dropdown", "options"),
    Input("edit_control", "geojson"),
    prevent_initial_call=True,
)
def mirror(x):

    # Prepare options
    choices = []
    rect_idx = 1
    if x is not None:
        for feat_idx, feature in enumerate(x["features"]):
            if feature["properties"]["type"] == "rectangle":
                coord = get_coord_from_feature(feature)
                choices.append(
                    {
                        "label": f"{rect_idx}: Bottom left coordinates: {coord}",
                        "value": f"{feat_idx}",
                    }
                )
                rect_idx += 1

    return choices


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


@app.callback(
    [Output("marker", "position")],
    [Input("polygon-dropdown", "value"), State("edit_control", "geojson")],
)
def add_marker(selected_polygon, polygons):
    """Generate masks, if not generated yet, when image is selected"""
    if selected_polygon == "None":
        return [[0, 0]]
    else:
        coord = get_polygon_coord(polygons, int(selected_polygon))
        bottom_left_coord = [coord[0][0], coord[1][0]]
        return [bottom_left_coord]


@app.callback(
    [Output("map", "children")],
    [
        Input("download_raster", "n_clicks"),
        State("map", "children"),
        State("polygon-dropdown", "value"),
        State("edit_control", "geojson"),
    ],
)
def download_raster_callback(n_clicks, cur_children, selected_polygon, polygons):

    ctx = callback_context.triggered
    if ctx[0]["prop_id"] == "download_raster.n_clicks":
        coord = get_polygon_coord(polygons, int(selected_polygon))

        foldername = get_next_folder_name(DATA_DIR)
        savedir = os.path.join(DATA_DIR, foldername)
        coords = get_raster_from_coord(
            lat=coord[0], long=coord[1], resolution=RESOLUTION, savedir=savedir
        )

        write_json(os.path.join(DATA_DIR, POLYGON_JSON_NAME), coords)

        img_paths = glob.glob(f"{savedir}/*.npy")
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

            cur_children.append(
                dl.ImageOverlay(
                    id="shown-image", opacity=1, url=png_path, bounds=converted_coord
                ),
            )
    else:
        coords = load_json(os.path.join(DATA_DIR, POLYGON_JSON_NAME))
        for key, tile_coord in coords.items():
            # tile_coord = coords[key]
            converted_coord = [
                [tile_coord["lat"][0], tile_coord["long"][0]],
                [tile_coord["lat"][1], tile_coord["long"][1]],
            ]
            cur_children.append(
                dl.ImageOverlay(
                    id="shown-image", opacity=1, url=key, bounds=converted_coord
                ),
            )

    return [cur_children]


app.run_server(debug=True, port=8888)
