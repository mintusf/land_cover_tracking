# land_cover_tracking
This is a Web APP for land cover tracking and simple analysis using satellite data.

Data is downloaded using integrated [sentinelhub](https://www.sentinel-hub.com/develop/api/) API.
Area of interest, as well as dates range, can be selected using the GUI.

Prediction is made using the model developed in [this repository](https://github.com/mintusf/land_cover_segmentation).

Predicted classes:
* shrublands
* urban
* agricultural
* barren
* snow
* water
* dense_forest
* open_forest
* wetlands
* grasslands

## Environment

#### Poetry
1. Install poetry with `pip install poetry`
2. Create environment with `poetry install`
3. Enter environment with `poetry shell`


#### Docker container
1. Create container with ./docker/setup_docker
2. Enter your container with `docker exec -it land_cover_app /bin/bash`


#### Raw environment
1. Install with pip install -r ./docker/requirements.txt


## App

1. Setup SentinelHub config file `sentinelhub_config.json`, according to [this guide](https://sentinelhub-py.readthedocs.io/en/latest/configure.html)
2. Download weigths from [here](https://drive.google.com/file/d/1REnApKRIkTvpRRLAbheBX_4OFJKQtp0N/view?usp=sharing) and put it to `weights` folder (more weigths coming)
3. To start, in your environment, please run `python app.py`.

Demo:
![DEMO](assets/DEMO.gif)
