"""
LoRA loaders for WeeLLM.

GenericLazyLoRALoader
    For any PEFT-style safetensors LoRA file – including massive ones (e.g. 9 GB).
    Parses only the safetensors header into RAM (~kilobytes); tensors are streamed
    from SSD to GPU one pair at a time during apply_to_module, then freed
    immediately.  Peak extra RAM is max(A_tensor_size, B_tensor_size) × 2 – a few
    MB at most, regardless of file size.

    Supported key prefixes in the file (auto-stripped):
        - transformer.transformer_blocks.N.*
        - transformer_blocks.N.*          (no transformer. prefix)
        - diffusion_model.transformer_blocks.N.*
        - base_model.model.transformer_blocks.N.*
"""

import os
import struct
import json
import logging

import numpy as np
import torch

logger = logging.getLogger("weellm")


# ---------------------------------------------------------------------------
# GenericLazyLoRALoader
# ---------------------------------------------------------------------------

class GenericLazyLoRALoader:
    """
    Memory-efficient PEFT-style LoRA loader for very large safetensors files.

    How it works
    ------------
    1. __init__: Open the file once, read the JSON header (kilobytes) and close.
       Build a ``lora_map`` dict: base_path → (A_key, B_key).
    2. apply_to_module: Open the file, iterate over matching pairs, seek to each
       tensor, read its bytes, push to GPU, compute delta, apply, free – all
       within a single file handle that lives only for the duration of the call.

    No mmap, no full-file load.  RAM overhead ≈ 2 × size_of_largest_tensor_pair.
    For LTX-2.5 at bf16 with rank 128 and dim 5120 that is ≈ 5 MB per call.

    Key normalisation (auto-detected, no config needed)
    ---------------------------------------------------
    The loader strips any of the following leading prefixes from LoRA keys:
        diffusion_model.
        base_model.model.
        transformer.
    so that ``base_path`` always starts with the unqualified layer name, e.g.
    ``transformer_blocks.47.attn1.to_q``.

    When apply_to_module is called with shard_name ``transformer_blocks.47`` (or
    ``transformer.transformer_blocks.47``), the matching base_paths are found and
    ``local_key = base_path[len(prefix):] + ".weight"`` is looked up in the
    module's named_parameters() dict.

    Skipped keys
    ------------
    If a LoRA key cannot be matched to a model parameter, a WARNING is emitted
    *once per unique key* (subsequent calls reuse ``_skipped_keys`` set).
    """

    _DTYPE_MAP: dict[str, tuple] = {
        "F64":  (np.float64,  None),
        "F32":  (np.float32,  None),
        "F16":  (np.float16,  None),
        "BF16": (np.uint16,   torch.bfloat16),
        "I64":  (np.int64,    None),
        "I32":  (np.int32,    None),
        "I16":  (np.int16,    None),
        "I8":   (np.int8,     None),
        "U8":   (np.uint8,    None),
    }

    def __init__(self, lora_path: str, scale: float = 1.0):
        self.lora_path    = lora_path
        self.scale        = scale
        self._skipped_keys: set[str] = set()

        # ----------------------------------------------------------------
        # Parse safetensors header without loading tensor data
        # ----------------------------------------------------------------
        with open(lora_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header_raw = f.read(n)
            self._header: dict = json.loads(header_raw.decode("utf-8"))

        self.data_base: int = 8 + n
        self._header.pop("__metadata__", None)
        self.keys: set[str] = set(self._header.keys())

        # ----------------------------------------------------------------
        # Build lora_pairs:  (A_file_key, B_file_key, targets)
        # ----------------------------------------------------------------
        from weellm.models.loras.lora_keymaps import build_lora_pairs
        self.lora_pairs, _ = build_lora_pairs(self.keys)

        logger.info(
            "[LoRA] Indexed %d LoRA weight pairs from: %s",
            len(self.lora_pairs),
            os.path.basename(lora_path),
        )

    # ------------------------------------------------------------------
    # Internal: read one tensor from disk (no mmap)
    # ------------------------------------------------------------------

    def _read_tensor(
        self,
        f,          # open file handle (binary read)
        key: str,
        device: torch.device,
    ) -> torch.Tensor:
        meta              = self._header[key]
        dtype_str: str    = meta["dtype"]
        shape: list[int]  = meta["shape"]
        start, end        = meta["data_offsets"]
        nbytes: int       = end - start

        np_dtype, torch_view_dtype = self._DTYPE_MAP.get(dtype_str, (None, None))
        if np_dtype is None:
            raise ValueError(f"[LoRA] Unsupported safetensors dtype '{dtype_str}' for key '{key}'")

        buf = bytearray(nbytes)
        f.seek(self.data_base + start)
        n_read = f.readinto(memoryview(buf))
        if n_read != nbytes:
            raise IOError(
                f"[LoRA] Short read for '{key}': expected {nbytes} bytes, got {n_read} bytes"
            )

        arr = np.frombuffer(buf, dtype=np_dtype)
        if shape:
            arr = arr.reshape(shape)

        t = torch.from_numpy(arr.copy())
        if torch_view_dtype is not None:
            t = t.view(torch_view_dtype)

        return t.to(device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public: apply matching LoRA tensors to one model shard
    # ------------------------------------------------------------------

    def apply_to_module(self, module: torch.nn.Module, shard_name: str):
        """
        Stream-apply LoRA deltas for *shard_name* into *module* (already in VRAM).

        Parameters
        ----------
        module     : The nn.Module for this shard (one transformer block, etc.).
        shard_name : Qualified name of the shard, e.g.:
                     "transformer_blocks.47"
                     "transformer.transformer_blocks.47"
        """
        # Normalise shard prefix – strip leading "transformer." so it matches
        # the normalised lora_map keys.
        prefix = shard_name
        if prefix.startswith("transformer."):
            prefix = prefix[len("transformer."):]
        if prefix and not prefix.endswith("."):
            prefix += "."

        params         = dict(module.named_parameters())
        applied_count  = 0
        skipped_count  = 0

        with open(self.lora_path, "rb") as f:
            for a_key, b_key, targets in self.lora_pairs:
                # Does this LoRA entry belong to the requested shard?
                matching_targets = []
                for t_key, slice_info in targets:
                    if t_key.startswith(prefix):
                        matching_targets.append((t_key[len(prefix):], slice_info))

                if not matching_targets:
                    continue

                valid_targets = []
                for local_key, slice_info in matching_targets:
                    param = params.get(local_key)
                    if param is None:
                        if local_key not in self._skipped_keys:
                            logger.warning(
                                "[LoRA] Key not found in model — SKIPPED: '%s'  (from A='%s')",
                                local_key,
                                a_key,
                            )
                            self._skipped_keys.add(local_key)
                        skipped_count += 1
                    else:
                        valid_targets.append((param, local_key, slice_info))

                if not valid_targets:
                    continue

                try:
                    a_t = self._read_tensor(f, a_key, valid_targets[0][0].device)   # float32 on GPU
                    b_t = self._read_tensor(f, b_key, valid_targets[0][0].device)   # float32 on GPU

                    with torch.no_grad():
                        alpha = None
                        alpha_key = a_key.rsplit('.lora_A', 1)[0] + '.alpha'
                        if alpha_key in self._header:
                            alpha_t = self._read_tensor(f, alpha_key, valid_targets[0][0].device)
                            alpha = alpha_t.item()
                            del alpha_t

                        rank = a_t.size(0)
                        
                        scale = self.scale
                        if alpha is not None:
                            scale = scale * (alpha / rank)
                            
                        delta = scale * (b_t @ a_t)
                        
                        for param, local_key, slice_info in valid_targets:
                            # Apply slicing if necessary
                            if slice_info is not None:
                                if slice_info[0] == "chunk_reorder":
                                    n_chunks = slice_info[1]
                                    order = slice_info[2]
                                    chunks = delta.chunk(n_chunks, dim=0)
                                    sub_delta = torch.cat([chunks[i] for i in order], dim=0)
                                else:
                                    split_idx, total_splits = slice_info
                                    chunks = delta.chunk(total_splits, dim=0)
                                    sub_delta = chunks[split_idx]
                            else:
                                sub_delta = delta
                                
                            param.data.add_(sub_delta.to(dtype=param.dtype))
                            applied_count += 1

                    # Release GPU memory immediately – do not accumulate across pairs
                    del a_t, b_t, delta
                    if 'sub_delta' in locals():
                        del sub_delta

                except Exception as exc:
                    logger.error(
                        "[LoRA] Failed to apply delta for '%s': %s",
                        valid_targets[0][1],
                        exc,
                        exc_info=True,
                    )
                    raise

        if applied_count > 0 or skipped_count > 0:
            logger.info(
                "[LoRA] Lazily applied %d LoRA tensors to '%s'  (%d skipped)",
                applied_count,
                shard_name,
                skipped_count,
            )
