import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger
from pytorch_lightning import Trainer

import cool_project
import cool_project.config as config
from cool_project.config import create_config
from cool_project.dataset import LitDataset

# from cool_project import Experiment
from cool_project.pipeline import LitModel

PROJECT_PATH = str(
    Path(os.path.dirname(cool_project.__file__)).parent.absolute()
)
load_dotenv(f"{PROJECT_PATH}/.env")
CONFIG_PATH = f"{PROJECT_PATH}/configs"

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    # seed_everything(7)
    config = create_config(f"{CONFIG_PATH}/starter.yml", verbose=True)

    dataset = LitDataset(**config.dataset_config.dict())
    dataset.prepare_data()
    dataset.setup()

    model = LitModel(
        model_config=config.model_config,
        optimizer_config=config.optimizer_config,
        lr_config=config.lr_config,
    )
    logger.info("Loaded")

    # exit()
    trainer = Trainer(**config.trainer_config.dict())
    trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())
