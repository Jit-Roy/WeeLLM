import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger("weellm")


def _extract_hub_repo_from_cache_path(path: Path):
    """
    If *path* is inside an HF Hub cache directory
    (e.g. .../models--org--repo/snapshots/<hash>/subfolder),
    return (repo_id, subfolder).  Returns (None, None) otherwise.
    """
    parts = path.parts
    for i, part in enumerate(parts):
        if part.startswith("models--"):
            # part looks like "models--HiDream-ai--HiDream-I1-Full"
            repo_id = part[len("models--"):].replace("--", "/", 1)
            # The subfolder is everything after .../snapshots/<hash>/
            try:
                snap_idx = parts.index("snapshots", i)
                subfolder_parts = parts[snap_idx + 2:]  # skip 'snapshots' and the hash
                subfolder = "/".join(subfolder_parts) if subfolder_parts else None
            except ValueError:
                subfolder = None
            return repo_id, subfolder
    return None, None


import threading
from contextlib import contextmanager

_override_local = threading.local()

@contextmanager
def override_weights_path(path: Union[str, Path, None]):
    """
    Temporarily overrides the path used by get_seeker() within the current thread.
    Useful for routing weights loading to a GGUF file without changing the directory
    path used for loading config.json.
    """
    old = getattr(_override_local, "weights_path", None)
    _override_local.weights_path = path
    try:
        yield
    finally:
        _override_local.weights_path = old

def get_seeker(model_dir: Union[str, Path], cache_to_ram: bool = False):
    """
    Factory function to return the appropriate tensor seeker.

    - **.gguf file path**: returns a GGUFSeeker that dequantizes weights on the
      fly using pure PyTorch — no custom CUDA compilation required.
    - **directory (default)**: returns SafetensorsRAMSeeker when cache_to_ram
      is True, otherwise SafetensorsDiskSeeker (original behaviour).
    """
    override = getattr(_override_local, "weights_path", None)
    if override is not None:
        model_dir = override
        
    model_dir_path = Path(model_dir)

    # ── GGUF: single-file path ending in .gguf ────────────────────────────────
    if model_dir_path.is_file() and model_dir_path.suffix.lower() == ".gguf":
        from weellm.gguf_seek import GGUFSeeker
        return GGUFSeeker(model_dir_path)
        
    model_dir_str = str(model_dir).replace("\\", "/")
    if model_dir_str.lower().endswith(".gguf") and not model_dir_path.exists():
        parts = model_dir_str.split("/")
        if len(parts) >= 3 and not Path(model_dir_str).is_absolute():
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:])
            logger.info("  [WeeLLM] GGUF file '%s' not found locally. Downloading from Hugging Face Hub (repo: %s)...", model_dir_str, repo_id)
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
            from weellm.gguf_seek import GGUFSeeker
            return GGUFSeeker(Path(downloaded_path))
        else:
            raise FileNotFoundError(f"GGUF file not found: {model_dir}")

    if not model_dir_path.exists():
        repo_id_str = str(model_dir).replace("\\", "/")
        is_hub_id   = repo_id_str.count("/") == 1 and not Path(repo_id_str).is_absolute()

        if is_hub_id:
            # Plain "namespace/repo" string — download the whole thing.
            logger.info("Directory '%s' not found. Attempting to download from Hugging Face Hub...", model_dir)
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(
                repo_id=repo_id_str,
                allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"],
            )
        else:
            # Absolute local path — try to recover a missing HF cache subfolder.
            repo_id, subfolder = _extract_hub_repo_from_cache_path(model_dir_path)
            if repo_id and subfolder:
                logger.info(
                    "Local subfolder '%s' not found. Downloading '%s' from repo '%s' ...",
                    model_dir_path, subfolder, repo_id,
                )
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[
                        f"{subfolder}/**",
                        f"{subfolder}/*",
                        f"{subfolder}/*.safetensors",
                        f"{subfolder}/*.json",
                    ],
                )
                # snapshot_download places files back into the existing cache;
                # the path should now exist.
                if not model_dir_path.exists():
                    raise FileNotFoundError(
                        f"Download attempted but directory still not found: '{model_dir_path}'"
                    )
            else:
                raise FileNotFoundError(
                    f"Local directory not found and path is not a valid Hub repo_id: '{model_dir}'"
                )
        model_dir_path = Path(model_dir)

    if cache_to_ram:
        from weellm.ram_seek import SafetensorsRAMSeeker
        return SafetensorsRAMSeeker(model_dir_path)
    else:
        from weellm.disk_seek import SafetensorsDiskSeeker
        return SafetensorsDiskSeeker(model_dir_path)
