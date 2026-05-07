from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable, Tuple


def _candidate_repo_paths(config=None) -> list[Path]:
    """Build candidate starVLA repo paths in priority order."""
    candidates: list[Path] = []
    if config is not None:
        repo = config.paths.get("starvla_repo", None)
        if repo:
            candidates.append(Path(str(repo)))

        ckpt = config.paths.get("starvla_ckpt", None)
        if ckpt:
            p = Path(str(ckpt)).resolve()
            for parent in [p] + list(p.parents):
                if parent.name == "starVLA":
                    candidates.append(parent)
                    break

    unique: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        rp = p.resolve()
        key = str(rp)
        if rp.exists() and key not in seen:
            seen.add(key)
            unique.append(rp)
    return unique


def _inject_paths(paths: Iterable[Path]) -> None:
    """Prepend candidate paths to sys.path for import resolution."""
    for p in paths:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)



def load_starvla_dataloader(config=None) -> Tuple[Callable, Callable]:
    """Load StarVLA dataloader entry points."""
    _inject_paths(_candidate_repo_paths(config))
    try:
        from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

        return collate_fn, get_vla_dataset
    except Exception as e:
        raise ImportError("Cannot import StarVLA dataloader.") from e
