import os

from config.default import get_cfg_from_file as get_cfg_app
from ai_engine.config.default import get_cfg_from_file as get_cfg_model
from ai_engine.models import get_model
from ai_engine.train_utils import get_optimizer, save_checkpoint
from ai_engine_wrapper.ai_engine_wrapper import ai_engine_infer

import rasterio as rio

def build_dummy_infer_raster(model_cfg, savepath):


def test_infer():

    model_config_path = os.path.join("ai_engine", "config", "tests.yml")
    app_cfg_path = os.path.join("config", "default.yml")
    checkpoint_save_path = os.path.join("tests", "ai_engine_wrapper", "test_checkpoint")
    infer_directory = os.path.join("tests", "ai_engine_wrapper", "test_masks")

    model_cfg = get_cfg_model(model_config_path)
    app_cfg = get_cfg_app(app_cfg_path)

    model = get_model(model_cfg, app_cfg.INFER.DEVICE)
    optimizer = get_optimizer(model, model_cfg)
    epoch = 1
    loss = 1.0

    save_checkpoint(model, epoch, optimizer, loss, model_cfg, checkpoint_save_path)

    ai_engine_infer(
        model_config_path, app_cfg, tile_path, checkpoint_save_path, infer_directory
    )
