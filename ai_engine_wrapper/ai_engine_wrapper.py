import torch
import os

from ai_engine.models import get_model
from ai_engine.dataset import get_dataloader
from ai_engine.config.default import get_cfg_from_file
from ai_engine.train_utils import load_checkpoint
from ai_engine.utils.utilities import get_gpu_count
from ai_engine.models.models_utils import (
    rename_ordered_dict_from_parallel,
    rename_ordered_dict_to_parallel,
)
from ai_engine.utils.io_utils import get_lines_from_txt
from ai_engine.utils.infer_utils import prepare_raster_for_inference
from ai_engine.infer import infer


def get_model_for_infer(model_cfg, app_cfg, checkpoint):
    model = get_model(model_cfg, app_cfg.INFER.DEVICE)

    if app_cfg.WORKERS > 0:
        torch.multiprocessing.set_start_method("spawn", force=True)

    _, weights, _, _, _ = load_checkpoint(checkpoint, app_cfg.INFER.DEVICE)
    if (
        get_gpu_count(model_cfg, mode="train") > 1
        and get_gpu_count(model_cfg, mode="test") == 1
    ):
        weights = rename_ordered_dict_from_parallel(weights)
    if (
        get_gpu_count(model_cfg, mode="train") == 1
        and get_gpu_count(model_cfg, mode="test") > 1
    ):
        weights = rename_ordered_dict_to_parallel(weights)
    model.load_state_dict(weights)

    return model


def ai_engine_infer(model_cfg_path, app_cfg, tile_path, checkpoint, destination):
    model_cfg = get_cfg_from_file(model_cfg_path)

    model = get_model_for_infer(model_cfg, app_cfg, checkpoint)

    samples_to_infer = []
    cropped_samples_paths = prepare_raster_for_inference(
        tile_path, crop_size=model_cfg.DATASET.SHAPE
    )
    samples_to_infer.extend(cropped_samples_paths)

    cropped_subgrids_list = os.path.join(
        os.path.split(tile_path)[0], "subgrids_list.txt"
    )
    with open(cropped_subgrids_list, "w") as f:
        for file in samples_to_infer:
            f.write(file + "\n")

    dataloader = get_dataloader(model_cfg, cropped_subgrids_list)

    infer(
        model,
        dataloader,
        output_types=["alphablended_raster"],
        destination=destination,
    )
