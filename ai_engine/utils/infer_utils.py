import glob
import os
from typing import List
from shutil import rmtree

from torch.utils.data import DataLoader
import torch
from torch.nn import Module

from ai_engine.utils.np_utils import (
    is_npy_cropped,
    crop_npy,
)
from ai_engine.utils.visualization_utils import (
    generate_save_alphablend,
)


def get_path_for_output(output_type, destination, name):
    output_destination = os.path.join(destination, output_type)
    os.makedirs(output_destination, exist_ok=True)
    extension = "tif" if "raster" in output_type else "png"
    name = os.path.splitext(os.path.split(name)[1])[0]
    alphablend_path = os.path.join(output_destination, name + f".{extension}")

    return alphablend_path


def prepare_npy_for_inference(path: str, crop_size: List[int]):
    paths_to_infer = []
    raster_folder, raster_file = os.path.split(path)

    if not is_npy_cropped(path, crop_size):
        paths_to_infer.append(path)
    else:

        raster_name = os.path.splitext(raster_file)[0]
        cropped_npy_directory = os.path.join(raster_folder, raster_name)

        if os.path.isdir(cropped_npy_directory):
            rmtree(cropped_npy_directory)
        os.makedirs(cropped_npy_directory)

        crop_npy(path, cropped_npy_directory, crop_size)

        paths_to_infer.extend(glob.glob(f"{cropped_npy_directory}/*.npy"))

    return paths_to_infer


def infer(
    model: Module,
    dataloader: DataLoader,
    destination: str,
):
    """Evaluates test dataset and saves predictions if needed

    Args:
        model (Module): Model to use for inference
        dataloader (DataLoader): Dataloader for inference
        destination (str): Path to save results

    Returns:
        dict: Generates and saves predictions in desired format
    """
    with torch.no_grad():
        model.eval()
        mask_config = dataloader.dataset.mask_config
        for batch in dataloader:
            inputs, names = batch["input"], batch["name"]

            # Forward propagation
            outputs = model(inputs)["out"]

            masks = torch.argmax(outputs, dim=1)

            for input_img, mask, name in zip(inputs, masks, names):

                output_path = get_path_for_output("alphablend", destination, name)
                generate_save_alphablend(input_img, mask, mask_config, output_path)
