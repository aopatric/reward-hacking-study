"""Thin Hydra entrypoint. All the actual wiring lives in src/corpus.py.

    uv run scripts/run_pilot.py
    uv run scripts/run_pilot.py model=qwen2.5-7b corpus.batch_size=64
"""

import json
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import src.config  # noqa: F401 -- registers Hydra structured-config schemas
from src.config import Config, set_seed
from src.corpus import run_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for _noisy_logger in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    set_seed(cfg.seed)

    run_dir = Path("runs") / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=run_dir / "config.yaml")

    summary = run_corpus(cfg, run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
