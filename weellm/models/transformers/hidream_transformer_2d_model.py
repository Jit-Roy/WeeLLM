"""
hidream_transformer_2d_model.py -- Hook-based layer streaming for HiDreamImageTransformer2DModel.

Streams `double_stream_blocks` and `single_stream_blocks`.
Keeps embeddings and final layers resident on GPU.
"""

import logging

logger = logging.getLogger("weellm")

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from weellm.seeker import get_seeker
from weellm.utils import clean_memory, report_memory


def _get_resident_keys(seeker) -> List[str]:
    """Return keys for embeddings and final layer."""
    prefixes = (
        "t_embedder.",
        "p_embedder.",
        "x_embedder.",
        "pe_embedder.",
        "final_layer.",
        "caption_projection.",
    )
    return [k for k in seeker.weight_map.keys() if any(k.startswith(p) for p in prefixes)]

def _get_double_block_keys(seeker, idx: int) -> List[str]:
    prefix = f"double_stream_blocks.{idx}."
    return [k for k in seeker.weight_map.keys() if k.startswith(prefix)]

def _get_single_block_keys(seeker, idx: int) -> List[str]:
    prefix = f"single_stream_blocks.{idx}."
    return [k for k in seeker.weight_map.keys() if k.startswith(prefix)]


class HiDreamImageTransformer2DModelStreamer:
    """
    Hook-based streaming wrapper for HiDreamImageTransformer2DModel.
    Loads each double/single block just-in-time during the forward pass, and evicts it right after.
    """

    def __init__(
        self,
        model: nn.Module,
        seeker,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.seeker = seeker
        self.device = device
        self.dtype = dtype

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tr_gpu_load")
        self._next_double_future = None
        self._next_double_idx: Optional[int] = None
        self._next_single_future = None
        self._next_single_idx: Optional[int] = None
        self._lock = threading.Lock()

        self._install_hooks(pinned_single_blocks=1)

    def _apply_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        for name, tensor in state_dict.items():
            if tensor.is_floating_point():
                set_module_tensor_to_device(self.model, name, self.device, value=tensor, dtype=self.dtype)
            else:
                set_module_tensor_to_device(self.model, name, self.device, value=tensor)

    def _evict_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        for name in state_dict.keys():
            set_module_tensor_to_device(self.model, name, "meta")

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_hooks(self, pinned_single_blocks: int = 1):
        for i, block in enumerate(self.model.double_stream_blocks):
            block._hidream_layer_idx = i
            block.register_forward_pre_hook(self._double_pre_hook)
            block.register_forward_hook(self._double_post_hook)

        num_single = len(self.model.single_stream_blocks)
        pinned_start_idx = max(0, num_single - pinned_single_blocks)

        for i, block in enumerate(self.model.single_stream_blocks):
            if i >= pinned_start_idx:
                keys = _get_single_block_keys(self.seeker, i)
                sd = self.seeker.get_tensors(keys, device=self.device, dtype=self.dtype)
                self._apply_state_dict(sd)
                continue

            block._hidream_layer_idx = i
            block.register_forward_pre_hook(self._single_pre_hook)
            block.register_forward_hook(self._single_post_hook)
            
        # Install hooks for resident modules so they evict *before* MoE blocks run
        resident_modules = [
            ("t_embedder.", self.model.t_embedder),
            ("p_embedder.", self.model.p_embedder),
            ("x_embedder.", self.model.x_embedder),
            ("pe_embedder.", self.model.pe_embedder),
            ("final_layer.", self.model.final_layer),
        ]
        if hasattr(self.model, "caption_projection") and self.model.caption_projection is not None:
            for i, proj in enumerate(self.model.caption_projection):
                resident_modules.append((f"caption_projection.{i}.", proj))
                
        for prefix, module in resident_modules:
            module._hidream_resident_prefix = prefix
            module.register_forward_pre_hook(self._resident_pre_hook)
            module.register_forward_hook(self._resident_post_hook)

    # ------------------------------------------------------------------
    # Resident Stream Hooks
    # ------------------------------------------------------------------

    def _resident_pre_hook(self, module: nn.Module, args):
        if not hasattr(self, "resident_sd"):
            return
        prefix = module._hidream_resident_prefix
        for name, tensor in self.resident_sd.items():
            if name.startswith(prefix):
                if tensor.is_floating_point():
                    set_module_tensor_to_device(self.model, name, self.device, value=tensor, dtype=self.dtype)
                else:
                    set_module_tensor_to_device(self.model, name, self.device, value=tensor)

    def _resident_post_hook(self, module: nn.Module, args, output):
        if not hasattr(self, "resident_sd"):
            return
        prefix = module._hidream_resident_prefix
        for name in self.resident_sd.keys():
            if name.startswith(prefix):
                set_module_tensor_to_device(self.model, name, "meta")

    # ------------------------------------------------------------------
    # Double Stream Hooks
    # ------------------------------------------------------------------

    def _double_pre_hook(self, module: nn.Module, args):
        idx = module._hidream_layer_idx
        keys = _get_double_block_keys(self.seeker, idx)

        with self._lock:
            if self._next_double_idx == idx and self._next_double_future is not None:
                sd = self._next_double_future.result()
                self._next_double_future = None
                self._next_double_idx = None
            else:
                sd = self.seeker.get_tensors(keys, device=self.device, dtype=self.dtype)

        self._apply_state_dict(sd)
        module._hidream_loaded_sd = sd

        # Prefetch next double block
        next_idx = idx + 1
        if next_idx < len(self.model.double_stream_blocks):
            next_keys = _get_double_block_keys(self.seeker, next_idx)
            with self._lock:
                self._next_double_future = self._executor.submit(
                    self.seeker.get_tensors, next_keys, self.device, self.dtype
                )
                self._next_double_idx = next_idx
        elif len(self.model.single_stream_blocks) > 0 and hasattr(self.model.single_stream_blocks[0], "_hidream_layer_idx"):
            # Prefetch first single block
            next_keys = _get_single_block_keys(self.seeker, 0)
            with self._lock:
                self._next_single_future = self._executor.submit(
                    self.seeker.get_tensors, next_keys, self.device, self.dtype
                )
                self._next_single_idx = 0

    def _double_post_hook(self, module: nn.Module, args, output):
        self._evict_state_dict(getattr(module, "_hidream_loaded_sd", {}))
        module._hidream_loaded_sd = {}
        return output

    # ------------------------------------------------------------------
    # Single Stream Hooks
    # ------------------------------------------------------------------

    def _single_pre_hook(self, module: nn.Module, args):
        idx = module._hidream_layer_idx
        keys = _get_single_block_keys(self.seeker, idx)

        with self._lock:
            if self._next_single_idx == idx and self._next_single_future is not None:
                sd = self._next_single_future.result()
                self._next_single_future = None
                self._next_single_idx = None
            else:
                sd = self.seeker.get_tensors(keys, device=self.device, dtype=self.dtype)

        self._apply_state_dict(sd)
        module._hidream_loaded_sd = sd

        # Prefetch next single block
        next_idx = idx + 1
        if next_idx < len(self.model.single_stream_blocks) and hasattr(self.model.single_stream_blocks[next_idx], "_hidream_layer_idx"):
            next_keys = _get_single_block_keys(self.seeker, next_idx)
            with self._lock:
                self._next_single_future = self._executor.submit(
                    self.seeker.get_tensors, next_keys, self.device, self.dtype
                )
                self._next_single_idx = next_idx

    def _single_post_hook(self, module: nn.Module, args, output):
        self._evict_state_dict(getattr(module, "_hidream_loaded_sd", {}))
        module._hidream_loaded_sd = {}
        return output

    @property
    def config(self):
        return self.model.config

    @property
    def max_seq(self):
        return self.model.max_seq
        
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __del__(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
    ) -> "HiDreamImageTransformer2DModelStreamer":
        from diffusers import HiDreamImageTransformer2DModel

        logger.info("Initializing SafetensorsLiveSeeker on HiDream Transformer weights ...")
        seeker = get_seeker(model_dir, cache_to_ram=cache_to_ram)

        logger.info("Instantiating HiDreamImageTransformer2DModel on meta device ...")
        config = HiDreamImageTransformer2DModel.load_config(model_dir)
        with init_empty_weights():
            model = HiDreamImageTransformer2DModel.from_config(config)
        model.eval()

        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        resident_keys = _get_resident_keys(seeker)
        logger.info("Holding resident tensors in CPU RAM (%d tensors) ...", len(resident_keys))

        resident_sd = seeker.get_tensors(resident_keys, device="cpu", dtype=dtype)

        # Do NOT apply them to the model here. Keep them on CPU and load during __call__

        clean_memory(device)

        logger.info(
            "  -> %d double stream and %d single stream blocks will stream on-demand.",
            len(model.double_stream_blocks),
            len(model.single_stream_blocks),
        )
        report_memory("After HiDream transformer init")

        streamer = cls(model, seeker, device, dtype)
        streamer.resident_sd = resident_sd
        return streamer
