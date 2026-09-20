"""Thin Hydra entrypoint. All the actual wiring lives in src/training.py."""

import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import src.config  # noqa: F401 -- registers Hydra structured-config schemas
from src.config import Config, set_seed
from src.training import build_trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for _noisy_logger in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    set_seed(cfg.seed)

    run_dir = Path("runs") / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=run_dir / "config.yaml")

    trainer = build_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
