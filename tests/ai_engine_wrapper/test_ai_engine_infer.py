import os
from shutil import rmtree

import torch
from config.default import get_cfg_from_file
from ai_engine.models import get_model
from utils.ai_engine_wrapper import ai_engine_infer

import numpy as np


def build_dummy_infer_npy(model_cfg, savepath):
    # Save raster
    input_size = model_cfg.DATASET.SHAPE
    band = np.random.randint(0, 30000, input_size, dtype=np.int16)

    bands_count = max(model_cfg.DATASET.INPUT.USED_CHANNELS) + 1
    img = np.concatenate(bands_count * [np.expand_dims(band, axis=2)], axis=2)
    np.save(savepath, img)


def test_infer():

    app_cfg_path = os.path.join("config", "default.yml")
    checkpoint_save_path = os.path.join("tests", "ai_engine_wrapper", "test_checkpoint")
    infer_directory = os.path.join("tests", "ai_engine_wrapper", "test_masks")
    tile_path = os.path.join("tests", "ai_engine_wrapper", "test_tile.npy")

    app_cfg = get_cfg_from_file(app_cfg_path)

    model = get_model(app_cfg, app_cfg.INFER.DEVICE)

    build_dummy_infer_npy(app_cfg, tile_path)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        checkpoint_save_path,
    )

    ai_engine_infer(app_cfg, tile_path, checkpoint_save_path, infer_directory)

    os.remove(checkpoint_save_path)
    os.remove(tile_path)
    os.remove(os.path.join("tests", "ai_engine_wrapper", "subgrids_list.txt"))
    rmtree(infer_directory)


test_infer()
