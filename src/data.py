"""Task-instance loading.

Original to this project. Kept out of env.py, which is attributed-port
territory (see that module's header) -- this is our own plumbing.

Returns raw (numbers, target) instances rather than rendered prompts: the
corpus renders each instance once per arm, so prompt construction has to
happen downstream of this, not inside it.
"""

from datasets import load_dataset

from src.config import EnvConfig


def load_task_instances(cfg: EnvConfig, seed: int, limit: int | None = None) -> list[dict]:
    dataset = load_dataset(cfg.dataset_name)["train"]
    n = min(limit or cfg.num_samples, len(dataset))
    dataset = dataset.shuffle(seed=seed).select(range(n))
    return [{"numbers": list(row["nums"]), "target": int(row["target"])} for row in dataset]
