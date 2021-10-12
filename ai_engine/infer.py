from typing import List

import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from ai_engine.utils.infer_utils import generate_outputs


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
