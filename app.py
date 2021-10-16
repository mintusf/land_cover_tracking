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
from ai_engine_wrapper.ai_engine_wrapper import ai_engine_infer

server = Flask(__name__)
app = Dash(server=server)
config = get_cfg_from_file("config/default.yml")

app.layout = html.Div(
    [
        html.H1(
            id="title",
            children="Land Cover Tracking",
            style={"textAlign": "center", "fontSize": 40},
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
                dl.Marker(
                    position=[0, 0],
                    id="marker_pred",
                ),
                dl.FeatureGroup([dl.EditControl(id="edit_control")]),
            ],
            style={
                "width": "100%",
                "height": "70vh",
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
            },
        ),
        html.Button(
            id="pred_button",
            children=html.H2(
                "Predict !", style={"textAlign": "center", "fontSize": 30}
            ),
            style={
                "display": "inline-block",
                "textalign": "center",
            },
        ),
        dcc.Dropdown(
            id="polygon-pred-dropdown",
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


@app.callback(
    Output("polygon-pred-dropdown", "options"),
    Input("map", "children"),
    prevent_initial_call=True,
)
def update_list_for_prediction(x):

    # Prepare options
    choices = []
    foldername = get_next_folder_name(DATA_DIR)
    if x is not None:
        for option in range(int(foldername)):
            coords = load_json(os.path.join(DATA_DIR, POLYGON_JSON_NAME))
            coord = coords[os.path.join(DATA_DIR, str(option), "tile_0.png")]
            coord_str = f"lat {coord['lat'][0]}, long {coord['long'][0]}"
            choices.append(
                {
                    "label": f"{option}: Bottom left coordinates: {coord_str}",
                    "value": f"{option}",
                }
            )

    return choices


@app.callback(
    [Output("marker_pred", "position")],
    [Input("polygon-pred-dropdown", "value"), State("edit_control", "geojson")],
)
def add_marker_pred(selected_polygon, polygons):
    """Generate masks, if not generated yet, when image is selected"""
    if selected_polygon == "None":
        return [[0, 0]]
    else:
        coords = load_json(os.path.join(DATA_DIR, POLYGON_JSON_NAME))
        coord = coords[os.path.join(DATA_DIR, str(selected_polygon), "tile_0.png")]
        top_right_coord = [coord["lat"][1], coord["long"][1]]
        return [top_right_coord]


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


import numpy as np


def merge_preds(polygon_id, tile_name, savedir):
    whole_img = cv2.imread(os.path.join(DATA_DIR, str(polygon_id), f"{tile_name}.png"))
    for pred_path in glob.glob(
        os.path.join(DATA_DIR, str(polygon_id), f"{tile_name}", "alphablend", "*.png")
    ):
        parts = os.path.splitext(os.path.split(pred_path)[1])[0].split("_")
        x_idx = int(parts[2])
        y_idx = int(parts[3])
        subgrid = cv2.imread(pred_path)
        whole_img[
            x_idx * 256 : (x_idx + 1) * 256, y_idx * 256 : (y_idx + 1) * 256, :
        ] = subgrid

    cv2.imwrite(savedir, whole_img)


@app.callback(
    [Output("map", "children")],
    [
        Input("download_raster", "n_clicks"),
        Input("pred_button", "n_clicks"),
        State("map", "children"),
        State("polygon-dropdown", "value"),
        State("polygon-pred-dropdown", "value"),
        State("edit_control", "geojson"),
    ],
)
def update_map(
    download_button,
    pred_button,
    cur_children,
    selected_polygon_download,
    selected_polygon_pred,
    polygons,
):

    ctx = callback_context.triggered
    if ctx[0]["prop_id"] == "download_raster.n_clicks":
        coord = get_polygon_coord(polygons, int(selected_polygon_download))

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
                    id="image", opacity=1, url=png_path, bounds=converted_coord
                ),
            )
    if ctx[0]["prop_id"] == "pred_button.n_clicks":
        for input_file in glob.glob(
            os.path.join(DATA_DIR, selected_polygon_pred, "*.npy")
        ):
            tile_name = os.path.splitext(os.path.split(input_file)[1])[0]
            ai_engine_infer(
                config,
                tile_path=input_file,
                checkpoint=config.INFER.WEIGHTS_PATH,
                destination=os.path.join(DATA_DIR, selected_polygon_pred, tile_name),
            )

            savedir = os.path.join(
                DATA_DIR, selected_polygon_pred, f"{tile_name}_pred.png"
            )
            merge_preds(selected_polygon_pred, tile_name, savedir=savedir)
            coords = load_json(os.path.join(DATA_DIR, POLYGON_JSON_NAME))
            tile_coord = coords[input_file.replace("npy", "png")]
            converted_coord = [
                [tile_coord["lat"][0], tile_coord["long"][0]],
                [tile_coord["lat"][1], tile_coord["long"][1]],
            ]
            cur_children.append(
                dl.ImageOverlay(
                    id="mask", opacity=1, url=savedir, bounds=converted_coord
                ),
            )

    else:
        coords = load_json(os.path.join(DATA_DIR, POLYGON_JSON_NAME))
        for key, tile_coord in coords.items():
            # tile_coord = coords[key]
            pred_path = key.replace(".png","_pred.png")
            if os.path.isfile(pred_path):
                url = pred_path
            else:
                url = key
            converted_coord = [
                [tile_coord["lat"][0], tile_coord["long"][0]],
                [tile_coord["lat"][1], tile_coord["long"][1]],
            ]
            cur_children.append(
                dl.ImageOverlay(id="mask", opacity=1, url=url, bounds=converted_coord),
            )

    return [cur_children]


app.run_server(debug=True, port=8888)
