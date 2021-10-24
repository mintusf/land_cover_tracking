from dash import Dash, callback_context
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_leaflet as dl
from flask import Flask, send_from_directory

from datetime import date
import os
from shutil import rmtree
import plotly.express as px

from config.default import get_cfg_from_file
from utils.dash_utils import (
    get_coord_from_feature,
    predict_action,
    download_action,
    get_polygon_coord,
    refresh_action,
    get_corner_coord,
    get_classes_count,
    get_top_labels,
    new_alpha_action,
    add_choice,
)

from utils.io_utils import (
    get_next_folder_name,
    load_json,
    write_json,
)
from utils.icons import (
    download_icon,
    pred_icon,
    analyze_icon,
    download_icon_url,
    pred_icon_url,
    analyze_icon_url,
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
                            icon=download_icon,
                            position=[0, 0],
                            id="marker",
                        ),
                        dl.Marker(
                            icon=pred_icon,
                            position=[0, 0],
                            id="marker_pred",
                        ),
                        dl.Marker(
                            icon=analyze_icon,
                            position=[0, 0],
                            id="marker_analyze",
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
                            children=[
                                html.H2(
                                    "Download raster",
                                    style={
                                        "display": "inline-block",
                                        "textAlign": "center",
                                        "fontSize": 25,
                                    },
                                ),
                                html.Img(
                                    src=download_icon_url,
                                    style={"display": "inline-block", "height": "5vh"},
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "textalign": "center",
                                "width": "55vh",
                                "margin-top": "5vh",
                            },
                        ),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="year-dropdown",
                                    placeholder="Year",
                                    value="2021",
                                    options=[
                                        {"label": year, "value": year}
                                        for year in range(2016, 2022)
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "width": "8vh",
                                        "height": "3vh",
                                        "fontSize": 20,
                                    },
                                ),
                                dcc.Dropdown(
                                    id="month-dropdown",
                                    placeholder="Season",
                                    value=1,
                                    options=[
                                        {"label": month, "value": month_id}
                                        for month, month_id in zip(
                                            [
                                                "JAN-MAR",
                                                "APR-JUN",
                                                "JUL-SEP",
                                                "OCT-DEC",
                                            ],
                                            [season_id for season_id in range(1, 5)],
                                        )
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "width": "9vh",
                                        "height": "3vh",
                                        "fontSize": 20,
                                    },
                                ),
                                dcc.Dropdown(
                                    id="polygon-dropdown",
                                    placeholder="Please choose the polygon",
                                    value="None",
                                    style={
                                        "display": "inline-block",
                                        "width": "38vh",
                                        "height": "3vh",
                                        "fontSize": 20,
                                    },
                                ),
                            ]
                        ),
                        html.Button(
                            id="pred_button",
                            children=[
                                html.H2(
                                    "Predict !",
                                    style={
                                        "display": "inline-block",
                                        "textAlign": "center",
                                        "fontSize": 25,
                                    },
                                ),
                                html.Img(
                                    src=pred_icon_url,
                                    style={"display": "inline-block", "height": "5vh"},
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "textalign": "center",
                                "width": "55vh",
                                "margin-top": "2vh",
                            },
                        ),
                        dcc.Dropdown(
                            id="polygon-pred-dropdown",
                            placeholder="Please choose the polygon for prediction",
                            value="None",
                            style={
                                "width": "100%",
                                "height": "3vh",
                                "fontSize": 20,
                            },
                        ),
                        html.Button(
                            id="analyze_button",
                            children=[
                                html.H2(
                                    "Analyze !",
                                    style={
                                        "display": "inline-block",
                                        "textAlign": "center",
                                        "fontSize": 25,
                                    },
                                ),
                                html.Img(
                                    src=analyze_icon_url,
                                    style={"display": "inline-block", "height": "5vh"},
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "textalign": "center",
                                "width": "55vh",
                                "margin-top": "2vh",
                            },
                        ),
                        html.Div(
                            children=[
                                html.H3(
                                    "Slider for prediction mask transparency",
                                    style={
                                        "textAlign": "center",
                                        "fontSize": 18,
                                    },
                                ),
                                dcc.Slider(
                                    id="ratio-slider",
                                    min=0,
                                    max=1,
                                    step=0.25,
                                    marks={
                                        0: "0",
                                        0.25: "0.25",
                                        0.5: "0.5",
                                        0.75: "0.75",
                                        1: "1",
                                    },
                                    value=0.5,
                                ),
                            ],
                            style={"backgroundColor": "white"},
                        ),
                        dcc.Dropdown(
                            id="polygon-analyze-dropdown",
                            placeholder="Please choose the polygon for prediction",
                            value="None",
                            style={
                                "width": "100%",
                                "height": "3vh",
                                "fontSize": 20,
                            },
                        ),
                        dcc.Graph(
                            id="graph",
                            style={
                                "width": "100%",
                                "height": "30vh",
                                "visibility": "hidden",
                            },
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "margin-left": "1vw",
                        "margin-left": "1vw",
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
STATS_JSON_NAME = config.STATS_JSON_NAME
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
                        "label": f"{rect_idx}: Coordinates: {coord}",
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
        coords = load_json(os.path.join(DATA_DIR, POLYGON_JSON_NAME))
        for option in range(int(foldername)):
            for season_id in range(1, 5):
                key = os.path.join(
                    DATA_DIR, str(option) + f"_s{season_id}", "tile_0.png"
                )
                if key not in coords:
                    continue
                coord = coords[key]
                add_choice(choices, coord, option, season_id)

    return choices


@app.callback(
    [Output("marker_pred", "position")],
    [Input("polygon-pred-dropdown", "value")],
)
def add_marker_pred(selected_polygon):
    """Generate masks, if not generated yet, when image is selected"""
    if selected_polygon == "None":
        return [[0, 0]]
    else:
        top_right_coord = get_corner_coord(
            selected_polygon, vertical="top", horizontal="right", config=config
        )
        return [top_right_coord]


@app.callback(
    [Output("marker_analyze", "position")],
    [Input("polygon-analyze-dropdown", "value")],
)
def add_marker_analyze(selected_polygon):
    """Generate masks, if not generated yet, when image is selected"""
    if selected_polygon == "None":
        return [[0, 0]]
    else:
        top_right_coord = get_corner_coord(
            selected_polygon, vertical="top", horizontal="right", config=config
        )
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
        top_right = [coord[0][1], coord[1][1]]
        return [top_right]


@app.callback(
    [Output("map", "children")],
    [
        Input("download_raster", "n_clicks"),
        Input("pred_button", "n_clicks"),
        Input("analyze_button", "n_clicks"),
        Input("ratio-slider", "value"),
        State("map", "children"),
        State("polygon-dropdown", "value"),
        State("polygon-pred-dropdown", "value"),
        State("polygon-analyze-dropdown", "value"),
        State("edit_control", "geojson"),
        State("year-dropdown", "value"),
        State("month-dropdown", "value"),
    ],
)
def update_map(
    download_button,
    pred_button,
    analyze_button,
    slider_value,
    cur_children,
    selected_polygon_download,
    selected_polygon_pred,
    selected_polygon_analyze,
    polygons,
    year,
    season,
):

    ctx = callback_context.triggered
    if ctx[0]["prop_id"] == "download_raster.n_clicks":

        paths, coords = download_action(
            polygons, selected_polygon_download, config, year, season
        )
        layer_name = "image"

    elif ctx[0]["prop_id"] == "pred_button.n_clicks":

        paths, coords = predict_action(config, selected_polygon_pred)
        layer_name = "mask"

    elif (
        ctx[0]["prop_id"] == "ratio-slider.value"
        or ctx[0]["prop_id"] == "analyze_button.n_clicks"
    ):
        if selected_polygon_analyze is None or selected_polygon_analyze == "None":
            return [cur_children]
        else:
            paths, coords = new_alpha_action(
                config, selected_polygon_analyze, 1 - slider_value
            )
            layer_name = "mask"

    else:
        paths, coords = refresh_action(config)
        layer_name = "loaded"

    for path, coord in zip(paths, coords):
        cur_children.append(
            dl.ImageOverlay(id=layer_name, opacity=1, url=path, bounds=coord),
        )

    return [cur_children]


@app.callback(
    Output("polygon-analyze-dropdown", "options"),
    Input("map", "children"),
    prevent_initial_call=True,
)
def update_list_for_analysis(x):

    # Prepare options
    choices = []
    foldername = get_next_folder_name(DATA_DIR)
    if x is not None:
        for option in range(int(foldername)):
            for season_id in range(1, 5):
                first_tile_path = os.path.join(
                    DATA_DIR, str(option) + f"_s{season_id}", "tile_0.png"
                )
                if os.path.isfile(first_tile_path.replace(".png", "_pred.png")):
                    coords = load_json(os.path.join(DATA_DIR, POLYGON_JSON_NAME))
                    coord = coords[first_tile_path]
                    add_choice(choices, coord, option, season_id)

    return choices


@app.callback(
    Output("graph", "figure"),
    Output("graph", "style"),
    [
        Input("analyze_button", "n_clicks"),
        State("polygon-analyze-dropdown", "value"),
        State("graph", "style"),
    ],
    prevent_initial_call=True,
)
def plot_stats(clicks, polygon_id, style):

    top_k = 5
    json_path = os.path.join(DATA_DIR, STATS_JSON_NAME)
    stats = load_json(json_path)
    if polygon_id not in stats:
        labels_counts = get_classes_count(polygon_id, config)
        stats[polygon_id] = labels_counts
        write_json(json_path, stats)
    else:
        labels_counts = stats[polygon_id]
    counts, top_labels = get_top_labels(labels_counts, k=top_k)
    cover = 100 * counts / counts.sum()
    fig = px.bar(
        x=top_labels,
        y=cover,
        range_y=[0, 100],
        title="Land cover [%]",
    )
    style["visibility"] = "visible"
    return fig, style


app.run_server(debug=True, port=8108)
