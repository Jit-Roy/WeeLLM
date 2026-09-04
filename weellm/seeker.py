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
def override_weights_path(path: Union[str, Path, None], subfolder: str = None):
    """
    Temporarily overrides the path used by get_seeker() within the current thread.
    Useful for routing weights loading to a GGUF file without changing the directory
    path used for loading config.json.
    """
    old_path = getattr(_override_local, "weights_path", None)
    old_sub  = getattr(_override_local, "subfolder", None)
    
    _override_local.weights_path = path
    _override_local.subfolder = subfolder
    try:
        yield
    finally:
        _override_local.weights_path = old_path
        _override_local.subfolder = old_sub

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
        
    # ── Single File Direct Hub Download ──────────────────────────────────────
    model_dir_str = str(model_dir).replace("\\", "/")
    
    # If it looks like a HuggingFace direct file path: "org/repo/path/to/file.ext"
    # and it doesn't exist locally as a relative path.
    if not model_dir_path.exists() and not model_dir_path.is_absolute():
        parts = model_dir_str.split("/")
        if len(parts) >= 3 and "." in parts[-1]:
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:])
            logger.info("  [WeeLLM] Hub file '%s' not found locally. Downloading from repo: %s...", model_dir_str, repo_id)
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model_dir_path = Path(downloaded_path)
            
    # ── GGUF Initialization ──────────────────────────────────────────────────
    if model_dir_path.is_file() and model_dir_path.suffix.lower() == ".gguf":
        from weellm.gguf_seek import GGUFSeeker
        return GGUFSeeker(model_dir_path)

    original_model_dir_str = str(model_dir).replace("\\", "/")
    explicit_subfolder = None
    if len(original_model_dir_str.split("/")) > 2 and not Path(original_model_dir_str).is_absolute():
        explicit_subfolder = "/".join(original_model_dir_str.split("/")[2:])

    if not model_dir_path.exists():
        parts = original_model_dir_str.split("/")
        is_hub_id = len(parts) >= 2 and not Path(original_model_dir_str).is_absolute()

        if is_hub_id:
            actual_repo_id = f"{parts[0]}/{parts[1]}"

            logger.info("Directory '%s' not found. Attempting to download from Hugging Face Hub (repo: %s)...", model_dir, actual_repo_id)
            from huggingface_hub import snapshot_download, HfApi
            try:
                files = HfApi().list_repo_files(repo_id=actual_repo_id)
                is_pipeline = "model_index.json" in files
            except Exception:
                is_pipeline = False

            # If user provided a subfolder in the string, use it. Otherwise, use the inferred one.
            target_subfolder = explicit_subfolder or getattr(_override_local, "subfolder", None)
            
            if is_pipeline and target_subfolder:
                allow_patterns = [
                    f"{target_subfolder}/*.safetensors",
                    f"{target_subfolder}/*.json",
                    f"{target_subfolder}/*.safetensors.index.json",
                    "model_index.json"
                ]
                logger.info("  Detected pipeline repo. Downloading ONLY subfolder '%s' ...", target_subfolder)
            else:
                allow_patterns = ["*.safetensors", "*.safetensors.index.json", "*.json"]

            model_dir = snapshot_download(
                repo_id=actual_repo_id,
                allow_patterns=allow_patterns,
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

    # Automatically append subfolder if the directory is a full pipeline repo
    final_target_subfolder = explicit_subfolder or getattr(_override_local, "subfolder", None)
    if final_target_subfolder and (model_dir_path / "model_index.json").exists():
        # Prevent double-appending if model_dir_path already contains the subfolder
        if not model_dir_path.name == final_target_subfolder:
            model_dir_path = model_dir_path / final_target_subfolder

    if cache_to_ram:
        from weellm.ram_seek import SafetensorsRAMSeeker
        return SafetensorsRAMSeeker(model_dir_path)
    else:
        from weellm.disk_seek import SafetensorsDiskSeeker
        return SafetensorsDiskSeeker(model_dir_path)
