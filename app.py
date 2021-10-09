from dash import Dash
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_leaflet as dl
from flask import Flask

from utils.dash_utils import get_coord_from_feature

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
            id="calc_cover",
            children=html.H2(
                "Calculate land cover", style={"textAlign": "center", "fontSize": 30}
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


@app.callback(
    Output("polygon-dropdown", "options"),
    Input("edit_control", "geojson"),
)
def mirror(x):
    if not x or not x["features"]:
        raise PreventUpdate

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


@app.callback(
    [Output("marker", "position")],
    [Input("polygon-dropdown", "value"), State("edit_control", "geojson")],
)
def add_marker(chosen_polygon, polygons):
    """Generate masks, if not generated yet, when image is selected"""

    if chosen_polygon == "None":
        raise PreventUpdate
    else:
        return [
            [
                polygons["features"][int(chosen_polygon)]["properties"]["_bounds"][0][
                    "lat"
                ],
                polygons["features"][int(chosen_polygon)]["properties"]["_bounds"][0][
                    "lng"
                ],
            ]
        ]


app.run_server(debug=True, port=8888)
