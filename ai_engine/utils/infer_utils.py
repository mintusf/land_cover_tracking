import glob
import os
from typing import List
from shutil import rmtree

from torch import Tensor
from torch.utils.data import DataLoader
import torch
from torch.nn import Module

from ai_engine.utils.raster_utils import (
    is_cropped,
    crop_raster,
    is_npy_cropped,
    crop_npy,
)
from ai_engine.utils.visualization_utils import (
    generate_save_alphablend,
    generate_save_alphablended_raster,
    generate_save_raster,
    generate_save_raw_raster,
)
from ai_engine.utils.utilities import split_sample_name


def get_save_path(
    name: str, destination: str, suffix: str, extention: str = "png"
) -> str:
    """Returns a path for an output sample.
        Creates directory if doesn't exist

    Args:
        name (str): Sample name
        destination (str): Root directory
        suffix (str): Suffix for file
    Returns:
        str: Save path
    """
    roi_folder, area, _ = split_sample_name(name)
    alphablend_folder = os.path.join(destination, roi_folder, area)
    if not os.path.isdir(alphablend_folder):
        os.makedirs(alphablend_folder)
    alphablend_path = os.path.join(alphablend_folder, f"{name}_{suffix}.{extention}")

    return alphablend_path


def get_path_for_output(output_type, destination, name, dataloader):
    output_destination = os.path.join(destination, output_type)
    os.makedirs(output_destination, exist_ok=True)
    extension = "tif" if "raster" in output_type else "png"
    if dataloader.dataset.mode == "infer":
        name = os.path.splitext(os.path.split(name)[1])[0]
        alphablend_path = os.path.join(output_destination, name + f".{extension}")
    else:
        alphablend_path = get_save_path(
            name, output_destination, output_type, extension
        )

    return alphablend_path


def generate_outputs(
    output_types: List[str],
    destination: str,
    input_img: Tensor,
    mask: Tensor,
    name: str,
    mask_config: dict,
    dataloader: DataLoader,
) -> None:
    """Generates and saves output images in formats specified by `output_types`

    Args:
        output_types (List[str]): List of output types.
                                  Currently supported:
                                  * alphablended (png alphablend)
                                  * alphablended_raster (tif alphablend)
                                  * raster (tif mask)
        destination (str): Root path to save outputs
        input_img (Tensor): Input img tensor
        mask (Tensor): Predicted mask tensor
        name (str): Sample name
        mask_config (dict): Mask config
        dataloader (DataLoader): Dataloader for samples
    """

    ref_raster_path = name

    for output_type in output_types:
        assert output_type in [
            "alphablend",
            "alphablended_raster",
            "raster",
            "raw_raster",
        ], f"Output type {output_type} not supported"
        output_path = get_path_for_output(output_type, destination, name, dataloader)

        if output_type == "alphablend":
            generate_save_alphablend(input_img, mask, mask_config, output_path)
        elif output_type == "alphablended_raster":
            generate_save_alphablended_raster(
                mask,
                input_img,
                mask_config,
                ref_raster_path,
                output_path,
            )
        elif output_type == "raw_raster":
            generate_save_raw_raster(
                input_img,
                ref_raster_path,
                output_path,
            )
        elif output_type == "raster":
            generate_save_raster(mask, mask_config, ref_raster_path, output_path)


def prepare_raster_for_inference(input_raster: str, crop_size: List[int]):
    paths_to_infer = []
    raster_folder, raster_file = os.path.split(input_raster)

    if not is_cropped(input_raster, crop_size):
        paths_to_infer.append(input_raster)
    else:

        raster_name = os.path.splitext(raster_file)[0]
        cropped_rasters_directory = os.path.join(raster_folder, raster_name)

        if os.path.isdir(cropped_rasters_directory):
            rmtree(cropped_rasters_directory)
        os.makedirs(cropped_rasters_directory)

        crop_raster(input_raster, cropped_rasters_directory, crop_size)

        paths_to_infer.extend(glob.glob(f"{cropped_rasters_directory}/*.tif"))

    return paths_to_infer


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
    output_types: List[str],
    destination: str,
):
    """Evaluates test dataset and saves predictions if needed

    Args:
        model (Module): Model to use for inference
        dataloader (DataLoader): Dataloader for inference
        output_types (List[str]): List of output types.
                                  Supported types:
                                    * alphablend (img and predicted mask)
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

                generate_outputs(
                    output_types,
                    destination,
                    input_img,
                    mask,
                    name,
                    mask_config,
                    dataloader,
                )
