from dash import Dash
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_leaflet as dl
from flask import Flask, send_from_directory
import glob

import os
import cv2

from utils.dash_utils import get_coord_from_feature
from utils.sentinel_api import get_raster_from_coord
from utils.io_utils import convert_sat_np_for_vis

server = Flask(__name__)
app = Dash(server=server)


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
                #     dl.ImageOverlay(
                #     id="shown-image", opacity=1, url='test\\tile_0.png', bounds=[(50.11707512706952, 20.028790064107014), (50.151130686729644, 20.09469931390529)]
                # ),
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

DATA_DIR = "new"


@server.route(f"/{DATA_DIR}/<path:path>")
def get_mask_url(path):
    """Serve a file from the upload directory."""
    img = send_from_directory(DATA_DIR, path, as_attachment=True, cache_timeout=0)
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
    prevent_initial_call=True,
)
def download_raster_callback(n_clicks, cur_children, selected_polygon, polygons):

    # if selected_polygon == "None":
    #     raise PreventUpdate

    coord = get_polygon_coord(polygons, int(selected_polygon))

    coords = get_raster_from_coord(coord[0], coord[1], DATA_DIR)

    img_paths = glob.glob(f"{DATA_DIR}/*.npy")
    for img_path in img_paths:
        img = convert_sat_np_for_vis(img_path)
        png_path = img_path.replace(".npy", ".png")
        cv2.imwrite(png_path, img)
        png_name = os.path.split(png_path)[1].replace(".png", "")

        cur_children.append(
            dl.ImageOverlay(
                id="shown-image", opacity=1, url=png_path, bounds=coords[png_name]
            ),
        )

    return [cur_children]


app.run_server(debug=True, port=8888)
