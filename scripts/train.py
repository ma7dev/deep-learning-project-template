import argparse
import os
from pathlib import Path

import torch
from loguru import logger
from pytorch_lightning import Trainer
from rich import pretty, traceback

import cool_project
from cool_project.config import create_config
from cool_project.dataset import LitDataset
from cool_project.pipeline import LitModel

pretty.install()
traceback.install()

PROJECT_PATH = str(
    Path(os.path.dirname(cool_project.__file__)).parent.absolute()
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default=f"{PROJECT_PATH}/configs/starter.yml",
        help="config path",
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = get_args()
    # seed_everything(7)

    logger.info("create config")
    config = create_config(args.config_path, verbose=True)

    logger.info("initialize dataset")
    dataset = LitDataset(**config.dataset_config.dict())
    logger.info("prepare dataset")
    dataset.prepare_data()
    logger.info("setup dataset")
    dataset.setup()

    logger.info("initialize model")
    model = LitModel(
        model_config=config.model_config,
        optimizer_config=config.optimizer_config,
        lr_config=config.lr_config,
    )

    logger.info("initialize trainer")
    trainer = Trainer(**config.trainer_config.dict())

    logger.info("fit model")
    trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())
