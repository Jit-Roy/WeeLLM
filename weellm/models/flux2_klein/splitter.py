"""
splitter.py -- One-time splitting of the Flux2 transformer's monolithic
safetensors checkpoint into per-layer shards.

The transformer comes as a single file (~7.2 GB):
    transformer/diffusion_pytorch_model.safetensors

We split it into:
    transformer/splitted_model/
        transformer_blocks.0.safetensors
        ...
        transformer_blocks.4.safetensors
        single_transformer_blocks.0.safetensors
        ...
        single_transformer_blocks.19.safetensors
        resident.safetensors          # all non-block weights (~390 MB)

This only needs to run once. Subsequent calls detect the shards and skip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


_DOUBLE_PREFIX = "transformer_blocks."
_SINGLE_PREFIX = "single_transformer_blocks."
_SHARD_DIR     = "splitted_model"
_DONE_SUFFIX   = ".done"


def _get_shard_path(shard_dir: Path, shard_name: str) -> Path:
    return shard_dir / f"{shard_name}.safetensors"


def _shard_exists(shard_dir: Path, shard_name: str) -> bool:
    return (
        _get_shard_path(shard_dir, shard_name).exists()
        and (shard_dir / f"{shard_name}.safetensors{_DONE_SUFFIX}").exists()
    )


def split_flux_transformer(
    transformer_dir: str | Path,
    shard_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """
    Split the monolithic transformer checkpoint into per-layer shards.

    Parameters
    ----------
    transformer_dir : str or Path
        Path to the ``transformer/`` subdirectory of the Flux2 Klein model.
        Must contain ``diffusion_pytorch_model.safetensors``.
    shard_dir : str or Path, optional
        Where to write the shards.  Defaults to
        ``<transformer_dir>/splitted_model/``.
    force : bool
        If True, re-split even if shards already exist.

    Returns
    -------
    shard_dir : Path
        Absolute path to the directory containing the per-layer shards.
    """
    transformer_dir = Path(transformer_dir)
    src = transformer_dir / "diffusion_pytorch_model.safetensors"
    if not src.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {src}")

    if shard_dir is None:
        shard_dir = transformer_dir / _SHARD_DIR
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Discover layer indices present in the checkpoint                  #
    # ------------------------------------------------------------------ #
    print("Scanning transformer checkpoint keys ...")
    with safe_open(str(src), framework="pt") as f:
        all_keys: List[str] = list(f.keys())

    double_indices: set[int] = set()
    single_indices: set[int] = set()
    resident_keys:  List[str] = []

    for k in all_keys:
        if k.startswith(_DOUBLE_PREFIX) and "single_transformer_blocks" not in k:
            double_indices.add(int(k.split(".")[1]))
        elif k.startswith(_SINGLE_PREFIX):
            single_indices.add(int(k[len(_SINGLE_PREFIX):].split(".")[0]))
        else:
            resident_keys.append(k)

    double_indices_sorted = sorted(double_indices)
    single_indices_sorted = sorted(single_indices)

    shard_names = (
        [f"transformer_blocks.{i}"        for i in double_indices_sorted]
        + [f"single_transformer_blocks.{i}" for i in single_indices_sorted]
        + ["resident"]
    )

    print(
        f"Found {len(double_indices_sorted)} double blocks, "
        f"{len(single_indices_sorted)} single blocks, "
        f"{len(resident_keys)} resident tensors."
    )

    # ------------------------------------------------------------------ #
    # 2. Skip if all shards already exist                                  #
    # ------------------------------------------------------------------ #
    if not force:
        if all(_shard_exists(shard_dir, n) for n in shard_names):
            print(f"Shards already exist at {shard_dir} -- skipping split.")
            return shard_dir

    # ------------------------------------------------------------------ #
    # 3. Stream source file and write one shard at a time                  #
    #    (safetensors supports random tensor access -- never loads 7.2 GB  #
    #     into RAM at once)                                                 #
    # ------------------------------------------------------------------ #
    print(f"Splitting {src.name} -> {shard_dir}")
    print("(This runs once and is skipped on future invocations)\n")

    with safe_open(str(src), framework="pt") as f:

        # --- double blocks ---
        for idx in tqdm(double_indices_sorted, desc="Double blocks"):
            shard_name = f"transformer_blocks.{idx}"
            if not force and _shard_exists(shard_dir, shard_name):
                continue
            prefix = f"{_DOUBLE_PREFIX}{idx}."
            state: Dict = {
                k: f.get_tensor(k)
                for k in all_keys
                if k.startswith(prefix) and "single_transformer_blocks" not in k
            }
            save_file(state, str(_get_shard_path(shard_dir, shard_name)))
            (shard_dir / f"{shard_name}.safetensors{_DONE_SUFFIX}").touch()

        # --- single blocks ---
        for idx in tqdm(single_indices_sorted, desc="Single blocks"):
            shard_name = f"single_transformer_blocks.{idx}"
            if not force and _shard_exists(shard_dir, shard_name):
                continue
            prefix = f"{_SINGLE_PREFIX}{idx}."
            state = {k: f.get_tensor(k) for k in all_keys if k.startswith(prefix)}
            save_file(state, str(_get_shard_path(shard_dir, shard_name)))
            (shard_dir / f"{shard_name}.safetensors{_DONE_SUFFIX}").touch()

        # --- resident tensors ---
        shard_name = "resident"
        if force or not _shard_exists(shard_dir, shard_name):
            state = {k: f.get_tensor(k) for k in resident_keys}
            save_file(state, str(_get_shard_path(shard_dir, shard_name)))
            (shard_dir / f"{shard_name}.safetensors{_DONE_SUFFIX}").touch()
            print(f"Resident tensors saved ({len(resident_keys)} keys).")

    print(f"\nSplit complete.  Shards written to: {shard_dir}")
    return shard_dir


def get_shard_path(shard_dir: Path, shard_name: str) -> Path:
    """Return the path to a named shard file."""
    return _get_shard_path(shard_dir, shard_name)
