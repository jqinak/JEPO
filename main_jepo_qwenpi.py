from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra

_THIS_DIR = Path(__file__).resolve().parent
_JEPO_ROOT = _THIS_DIR
for _p in (_JEPO_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

@hydra.main(config_path="config", config_name="jepo_qwenpi", version_base=None)
def main(config):
    run_jepo(config)


def run_jepo(config):
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    use_ray = bool(config.runtime.get("use_ray", False))
    if use_ray:
        import ray

        if not ray.is_initialized():
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN",
                    }
                }
            )

    from jepo.trainer import JEPORayTrainer

    trainer = JEPORayTrainer(config=config)
    trainer.fit()


if __name__ == "__main__":
    main()
