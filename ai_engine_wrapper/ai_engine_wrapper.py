import torch
import os

from ai_engine.models import get_model
from ai_engine.dataset import get_dataloader
from ai_engine.train_utils import load_checkpoint
from ai_engine.models.models_utils import (
    rename_ordered_dict_from_parallel,
    rename_ordered_dict_to_parallel,
)
from ai_engine.utils.infer_utils import prepare_npy_for_inference
from ai_engine.utils.infer_utils import infer


def get_model_for_infer(model_cfg, app_cfg, checkpoint):
    model = get_model(model_cfg, app_cfg.INFER.DEVICE)

    if app_cfg.INFER.WORKERS > 0:
        torch.multiprocessing.set_start_method("spawn", force=True)

    weights = load_checkpoint(checkpoint, app_cfg.INFER.DEVICE)
    if False:
        weights = rename_ordered_dict_from_parallel(weights)
    if False:
        weights = rename_ordered_dict_to_parallel(weights)
    model.load_state_dict(weights)

    return model


def ai_engine_infer(app_cfg, tile_path, checkpoint, destination):

    model = get_model_for_infer(app_cfg, app_cfg, checkpoint)

    samples_to_infer = []
    cropped_samples_paths = prepare_npy_for_inference(
        tile_path, crop_size=app_cfg.DATASET.SHAPE
    )
    samples_to_infer.extend(cropped_samples_paths)

    cropped_subgrids_list = os.path.join(
        os.path.split(tile_path)[0], "subgrids_list.txt"
    )
    with open(cropped_subgrids_list, "w") as f:
        for file in samples_to_infer:
            f.write(file + "\n")

    dataloader = get_dataloader(app_cfg, cropped_subgrids_list)

    infer(
        model,
        dataloader,
        output_types=["alphablend"],
        destination=destination,
    )
