from dash import Dash, callback_context
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_leaflet as dl
from flask import Flask, send_from_directory
import glob

import os
from shutil import rmtree

from config.default import get_cfg_from_file
from utils.dash_utils import (
    get_coord_from_feature,
    predict_action,
    download_action,
    get_polygon_coord,
    refresh_action,
)

from utils.io_utils import (
    get_next_folder_name,
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
            style={
                "textAlign": "center",
                "fontSize": 40,
                "margin-top": "0vw",
                "margin-bottom": "0vw",
            },
        ),
        # Setup a map with the edit control.
        html.Div(
            children=[
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
                        "width": "70%",
                        "height": "91vh",
                        "margin": "auto",
                        "display": "inline-block",
                        "position": "relative",
                    },
                    id="map",
                ),
                html.Div(
                    children=[
                        html.Button(
                            id="download_raster",
                            children=html.H2(
                                "Download raster",
                                style={"textAlign": "center", "fontSize": 25},
                            ),
                            style={
                                "display": "inline-block",
                                "textalign": "center",
                                "width": "50vh",
                                "margin-top": "5vh",
                            },
                        ),
                        dcc.Dropdown(
                            id="polygon-dropdown",
                            placeholder="Please choose the polygon",
                            value="None",
                            style={
                                "width": "50vh",
                                "height": "4vh",
                                "fontSize": 20,
                            },
                        ),
                        html.Button(
                            id="pred_button",
                            children=html.H2(
                                "Predict !",
                                style={"textAlign": "center", "fontSize": 25},
                            ),
                            style={
                                "display": "inline-block",
                                "textalign": "center",
                                "width": "50vh",
                                "margin-top": "5vh",
                            },
                        ),
                        dcc.Dropdown(
                            id="polygon-pred-dropdown",
                            placeholder="Please choose the polygon",
                            value="None",
                            style={
                                "width": "100%",
                                "height": "4vh",
                                "fontSize": 20,
                            },
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "margin-left": "3vw",
                        "position": "absolute",
                        "height": 200,
                    },
                ),
            ],
            style={"backgroundColor": "grey"},
        ),
    ]
)

DATA_DIR = config.DATA_DIR
POLYGON_JSON_NAME = config.POLYGON_JSON_NAME
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
                        "label": f"{rect_idx}: Bottom left coord: {coord}",
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
            coord_str = f"lat {coord['lat'][0]:.2f}, long {coord['long'][0]:.2f}"
            choices.append(
                {
                    "label": f"{option}: Top left coord: {coord_str}",
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

        paths, coords = download_action(polygons, selected_polygon_download, config)
        layer_name = "image"

    if ctx[0]["prop_id"] == "pred_button.n_clicks":

        paths, coords = predict_action(config, selected_polygon_pred)
        layer_name = "mask"

    else:
        paths, coords = refresh_action(config)
        layer_name = "loaded"

    for path, coord in zip(paths, coords):
        cur_children.append(
            dl.ImageOverlay(id=layer_name, opacity=1, url=path, bounds=coord),
        )

    return [cur_children]


app.run_server(debug=True, port=8888)
