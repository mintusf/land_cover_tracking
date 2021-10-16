from typing import Dict, Tuple, Union

import cv2
import numpy as np

from torch import Tensor

from ai_engine.utils.np_utils import (
    convert_np_for_vis,
)


def apply_single_mask(
    image: np.array, mask: np.array, color: tuple, alpha: float = 0.6
) -> np.array:
    """A method to generate visualization of masks
    Args:
        image (np.array): Input image
        mask (np.array): Mask
        color (tuple): Color of mask (R, G, B)
        alpha (float, optional): Non-transparency of mask. Defaults to 0.6.
    Returns:
        np.array: Image with mask visualization
    """
    out = image.copy()
    for c in range(3):
        out[:, :, c] = np.where(
            mask != 0, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c]
        )
    return out


def create_alphablend(
    img: np.array,
    mask: np.array,
    alpha: float,
    colors_dict: dict,
    class2label: Union[None, Dict[int, str]] = None,
) -> np.array:
    """A method to create alphablend image

    Args:
        img (np.array): Input image
        mask (np.array): Mask
        alpha (float): Alpha value
        colors_dict (dict): Dictionary matching class id to color
        class2label (dict): Dictionary matching class id to label

    Returns:
        np.array: Alphablend image
    """

    x_pos = 30
    y_pox = 30
    for class_int, color in colors_dict.items():
        class_mask = np.where(mask == class_int, 1, 0)
        img = apply_single_mask(img, class_mask, color, alpha)
        if class_mask.sum() > 100 and class2label is not None:
            cv2.putText(
                img,
                class2label[class_int],
                (x_pos, y_pox),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            y_pox += 30

    return img


def generate_save_alphablend(
    input_img: Tensor,
    mask: Tensor,
    mask_config: dict,
    alphablend_path: str,
):
    """Generates and saves alphablend

    Args:
        input_img (Tensor): Input img tensor
        mask (Tensor): Predicted mask tensor
        name (str): Sample name
        mask_config (dict): Mask config
        alphablend_destination (str): Root path to save alphablend
    """
    input_img, mask = prepare_tensors_for_vis(input_img, mask)
    alpha = mask_config["alpha"]
    colors_dict = mask_config["colors"]
    class2label = mask_config["class2label"]
    alphablended = create_alphablend(input_img, mask, alpha, colors_dict, class2label)
    alphablended = cv2.cvtColor(alphablended, cv2.COLOR_BGR2RGB)
    cv2.imwrite(alphablend_path, alphablended)


def prepare_tensors_for_vis(
    input_img: Tensor, mask: Union[None, Tensor]
) -> Union[np.array, Tuple[np.array, np.array]]:
    """Prepares input and mask for visualization

    Args:
        input_img (Tensor): Input img tensor
        mask (Tensor): Predicted mask tensor

    Returns:
        Tuple[np.array, np.array]: Input and mask for visualization
    """
    input_img = input_img.cpu().numpy()
    input_img = input_img[(2, 1, 0), :, :]
    input_img = convert_np_for_vis(input_img)

    if mask is None:
        return input_img
    else:
        mask = mask.cpu().numpy()
        return input_img, mask
