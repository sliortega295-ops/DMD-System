"""Dedicated Grid-1000 MiniMax-H3 DMD training entrypoint."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from loguru import logger

from . import model as _model  # noqa: F401
from . import trainer as _trainer  # noqa: F401
from . import grid_adaln as _grid_adaln  # noqa: F401
from . import grid_timestep_compat as _grid_timestep_compat  # noqa: F401
from . import grid_contract as _grid_contract  # noqa: F401
from .distributed import init_distributed_a100

from lightx2v_train.data import build_data, prepare_data
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import cleanup_distributed, load_config, setup_logger
from lightx2v_train.runtime.distributed import get_rank
from lightx2v_train.trainers import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Grid-1000 MiniMax-H3 DMD")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _seed_rank(config) -> int:
    seed = int(config.get("training", {}).get("seed", 20260817)) + int(get_rank())
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def main():
    args = parse_args()
    config = load_config(args.config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    init_distributed_a100(config)
    seed = _seed_rank(config)
    setup_logger(config)
    logger.info("[h3-a100][grid1000] seed={} rank={}", seed, get_rank())
    try:
        prepare_data(config)
        model = build_model(config)
        model.load_components()
        dataloader_train = build_data(config, train_or_val="train")
        trainer = build_trainer(config)
        trainer.set_model(model)
        trainer.set_data(dataloader_train, None)
        trainer.train()
    except Exception:
        logger.exception("Grid-1000 H3 A100 training failed")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
