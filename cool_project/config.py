import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torchvision.transforms as T
import yaml
from devtools import debug
from loguru import logger
from pydantic import (
    BaseModel,
    PositiveFloat,
    PositiveInt,
    StrictBool,
    ValidationError,
)
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import cool_project

PROJECT_PATH = str(
    Path(os.path.dirname(cool_project.__file__)).parent.absolute()
)


def get_exp_path_and_run_name(
    output_path: str, exp_name: str
) -> Tuple[str, str, str]:
    random_str = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=5)
    )
    today = datetime.today().strftime("%Y-%m-%d")
    curr_time = datetime.today().strftime("%H-%M")
    run_name = f"{curr_time}-{exp_name}-{random_str}"
    exp_path = f"{output_path}/{today}/{run_name}"
    return exp_path, run_name, today


def create_log_dir(output_path: str, exp_path: str, today: str) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(f"{output_path}/{today}"):
        os.makedirs(f"{output_path}/{today}")
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        os.makedirs(f"{exp_path}/ckpts")
        os.makedirs(f"{exp_path}/optims")
        os.makedirs(f"{exp_path}/figs")
        os.makedirs(f"{exp_path}/logs")
    else:
        raise Exception(f"Experiment path {exp_path} already exists")


def exp(output_path: str, exp_name: str) -> Tuple[str, str]:
    exp_path, run_name, today = get_exp_path_and_run_name(
        output_path, exp_name
    )
    create_log_dir(output_path, exp_path, today)
    return exp_path, run_name


def get_transform(is_train: bool = False) -> T.Compose:
    transform = []
    if is_train:
        transform.extend(
            [
                T.RandomRotation(5),
            ]
        )
    transform.extend(
        [
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return T.Compose(transform)


class DatasetConfig(BaseModel):
    data_path: str = f"{PROJECT_PATH}/data"
    batch_size: PositiveInt = 256
    num_workers: PositiveInt = 4
    transform_fn: Any = get_transform


class ModelConfig(BaseModel):
    in_channels: PositiveInt = 1 * 28 * 28
    hidden_size: PositiveInt = 64
    num_classes: PositiveInt = 10


class OptimizerConfig(BaseModel):
    cls: str = "Adam"
    kwargs: Dict[str, float] = {
        "lr": 2e-4,
    }


class LRConfig(BaseModel):
    cls: str = "OneCycleLR"
    max_lr: PositiveFloat = 0.1
    interval: str = "step"


class ExperimentConfig(BaseModel):
    exp_name: str = "baseline"
    exp_path: str = ""
    run_name: str = ""
    output_path = f"{PROJECT_PATH}/output"


class TrainerConfig(BaseModel):
    devices: List[PositiveInt] = [0]
    fast_dev_run: StrictBool = False
    overfit_batches: PositiveFloat = 0.0
    max_epochs: PositiveInt = 10
    accelerator: str = "gpu"
    num_sanity_val_steps: PositiveInt = 0
    check_val_every_n_epoch: PositiveInt = 1
    accumulate_grad_batches: PositiveInt = 1
    log_every_n_steps: PositiveInt = 10
    callbacks: List[str] = []
    logger: List[str] = []


class Config(BaseModel):
    project_name: str = "cifar10_experiments"
    dataset_config: DatasetConfig = DatasetConfig()
    model_config: ModelConfig = ModelConfig()
    optimizer_config: OptimizerConfig = OptimizerConfig()
    lr_config: LRConfig = LRConfig()
    experiment_config: ExperimentConfig = ExperimentConfig()
    trainer_config: TrainerConfig = TrainerConfig()


def get_callbacks(
    config: Config, callbacks_dict: Dict[Any, Any] = {}
) -> List[Any]:
    callbacks: List[Any] = []
    if "tqdm" in callbacks_dict and callbacks_dict["tqdm"]:
        callbacks.append(
            TQDMProgressBar(
                refresh_rate=callbacks_dict["tqdm"]["refresh_rate"]
                if "refresh_rate" in callbacks_dict["tqdm"]
                else 10
            )
        )
    if "checkpoint" in callbacks_dict and callbacks_dict["checkpoint"]:
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"{config.experiment_config.exp_path}/ckpts",
                monitor=callbacks_dict["checkpoint"]["monitor"]
                if "monitor" in callbacks_dict["checkpoint"]
                else "val/epoch/loss",
            )
        )
    return callbacks


def get_logger(config: Config, logger_dict: Dict[Any, Any] = {}) -> List[Any]:
    logger: List[Any] = []
    if "wandb" in logger_dict and logger_dict["wandb"]:
        logger.append(
            WandbLogger(
                project=config.project_name,
                name=config.experiment_config.run_name,
                save_dir=config.experiment_config.output_path,
            )
        )
    if "tensorboard" in logger_dict and logger_dict["tensorboard"]:
        logger.append(
            TensorBoardLogger(
                save_dir=f"{config.experiment_config.output_path}/tb_logs",
                name=config.experiment_config.run_name,
            )
        )
    return logger


def create_config(config_path: str, verbose: bool = False) -> Config:
    try:
        with open(Path(config_path), "r") as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)

        if config_dict is None:
            config_dict = {}

        config = Config(**config_dict)

        (
            config.experiment_config.exp_path,
            config.experiment_config.run_name,
        ) = exp(
            config.experiment_config.output_path,
            config.experiment_config.exp_name,
        )
        if "callbacks" in config_dict:
            config.trainer_config.callbacks = get_callbacks(
                config, config_dict["callbacks"]
            )
        if "logger" in config_dict:
            config.trainer_config.logger = get_logger(
                config, config_dict["logger"]
            )
    except ValidationError as e:
        raise Exception(f"Invalid config: {e}") from e

    if verbose:
        logger.info("verbose")
        debug(f"config_path: {config_path}")
        debug(config_dict)
        debug(config)

    logger.info("logging into file")
    with open(
        f"{config.experiment_config.exp_path}/logs/config.log", "w"
    ) as f:
        yaml.dump(f"config_path: {config_path}")
        yaml.dump(config_dict, f)
        yaml.dump(config.dict(), f)
    return config
