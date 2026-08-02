"""
text_encoder_streamer.py -- Memory-frugal streaming of the Qwen3 text encoder.

Pipeline strategy (same as FluxStreamer):
  Background thread: disk -> GPU directly via load_file(device='cuda')
  Main thread: sync event, apply weights, run forward pass

Per-layer load: ~7ms disk + ~30ms H2D (202MB) = ~37ms in background.
Per-layer compute: ~300-400ms on GPU.
=> Load is fully hidden inside previous layer's forward pass.

The Flux2 pipeline uses a SPECIFIC encoding strategy:
  - Apply chat template to the prompt
  - Run through Qwen3ForCausalLM with output_hidden_states=True
  - Collect hidden states from layers 9, 18, and 27
  - Stack and reshape: (B, seq, 3*hidden_dim) = (B, seq, 7680)

We use the SAME hook-based approach as FluxStreamer:
  - A forward_pre_hook on each decoder layer loads its shard from disk to GPU
  - A forward_hook evicts it back to meta after the forward call
  - The full model.model.forward() is called normally so all internal logic
    (RoPE, causal masks, etc.) runs correctly without any manual plumbing

Resident modules kept on GPU (~45 MB total):
  - model.embed_tokens  (~22 MB)
  - model.rotary_emb    (~1 KB, tiny buffers)
  - model.norm          (~5 KB)

Peak VRAM during encoding: ~45 MB resident + ~220 MB one Qwen3 layer = ~265 MB
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Dict

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from weellm.core.utils import clean_memory, report_memory


# Layers from which the Flux2 pipeline extracts hidden states.
_DEFAULT_EXTRACT_LAYERS = (9, 18, 27)


def _load_layer_to_cuda(path: str, device: str):
    """
    Load a text encoder layer shard directly from disk to GPU.
    Runs in a background thread; returns (gpu_tensors, cuda_event) so the
    main thread can synchronise cheaply without a full device sync.
    """
    from typing import Tuple
    import torch
    from safetensors.torch import load_file
    sd = load_file(path, device=device)
    event = torch.cuda.Event()
    event.record()
    return sd, event


class StreamingQwen3TextEncoder:
    """
    Hook-based streaming of the Qwen3 text encoder.

    Uses the same pre/post hook pattern as FluxStreamer so that the full
    model.model.forward() runs without any manual internal-API plumbing.
    Captures intermediate hidden states at the configured layer indices.

    Parameters
    ----------
    text_encoder_dir : str or Path
    tokenizer_dir    : str or Path
    device           : str
    dtype            : torch.dtype
    extract_layers   : tuple[int]   Qwen3 layers to capture (default: (9,18,27))
    max_length       : int          Max tokenised prompt length (default: 512)
    """

    def __init__(
        self,
        text_encoder_dir: str | Path,
        tokenizer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        extract_layers: tuple = _DEFAULT_EXTRACT_LAYERS,
        max_length: int = 512,
    ):
        self.text_encoder_dir = Path(text_encoder_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.device = device
        self.dtype = dtype
        self.extract_layers = tuple(sorted(extract_layers))
        self.max_length = max_length

        self._shard_dir: Optional[Path] = None
        self._model: Optional[nn.Module] = None
        self._tokenizer = None
        self._num_layers: int = 0
        self._initialized = False

        # Hook state: captured hidden states from target layers
        self._captured: Dict[int, torch.Tensor] = {}
        self._hook_handles = []

        # Direct GPU load pipeline (same pattern as FluxStreamer)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._next_future = None           # Future[(gpu_tensors, cuda_event)]
        self._next_future_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_initialized(self):
        if self._initialized:
            return
        print("Initialising streaming Qwen3 text encoder ...")
        self._setup_shard_dir()
        self._load_model_skeleton()
        self._load_tokenizer()
        self._load_resident_modules()
        self._install_hooks()
        # Background GPU loader (1 worker -- no GPU contention)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="te_gpu_load"
        )
        self._initialized = True
        print("Text encoder ready (streaming).")

    # ------------------------------------------------------------------
    # Shard splitting
    # ------------------------------------------------------------------

    def _setup_shard_dir(self):
        """Split text encoder safetensors into per-layer shards (once)."""
        shard_dir = self.text_encoder_dir / "splitted_model"
        shard_dir.mkdir(parents=True, exist_ok=True)

        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        n_layers = config.num_hidden_layers
        self._num_layers = n_layers

        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        # Required shards (lm_head is optional when weights are tied)
        required_shards = (
            ["model.embed_tokens", "model.norm"]
            + [f"model.layers.{i}" for i in range(n_layers)]
            + ([] if tie_word_embeddings else ["lm_head"])
        )

        all_done = all(
            (shard_dir / f"{name}.safetensors").exists()
            and (shard_dir / f"{name}.safetensors.done").exists()
            for name in required_shards
        )

        if all_done:
            print(f"  Text encoder shards already exist at {shard_dir}")
            self._shard_dir = shard_dir
            return

        print(f"  Splitting text encoder ({n_layers} layers) -- one-time operation ...")
        self._split_text_encoder(shard_dir, required_shards)
        self._shard_dir = shard_dir

    def _split_text_encoder(self, shard_dir: Path, shard_names: List[str]):
        from safetensors import safe_open
        from safetensors.torch import save_file
        from collections import defaultdict
        from tqdm import tqdm
        import json

        index_path = self.text_encoder_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
        else:
            src = self.text_encoder_dir / "model.safetensors"
            with safe_open(str(src), framework="pt") as f:
                weight_map = {k: "model.safetensors" for k in f.keys()}

        for shard_name in tqdm(shard_names, desc="Splitting text encoder"):
            out_path = shard_dir / f"{shard_name}.safetensors"
            done_path = shard_dir / f"{shard_name}.safetensors.done"
            if out_path.exists() and done_path.exists():
                continue

            prefix = shard_name + "."
            layer_keys = [k for k in weight_map if k.startswith(prefix)]
            if not layer_keys:
                continue

            by_file: Dict[str, List[str]] = defaultdict(list)
            for k in layer_keys:
                by_file[weight_map[k]].append(k)

            state = {}
            for fname, keys in by_file.items():
                with safe_open(str(self.text_encoder_dir / fname), framework="pt") as f:
                    for k in keys:
                        state[k] = f.get_tensor(k)

            save_file(state, str(out_path))
            done_path.touch()

        print(f"  Text encoder split complete -> {shard_dir}")

    # ------------------------------------------------------------------
    # Model skeleton
    # ------------------------------------------------------------------

    def _load_model_skeleton(self):
        """Instantiate the Qwen3 model on meta (zero VRAM)."""
        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        with init_empty_weights():
            self._model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        self._model.eval()

    def _load_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_dir), trust_remote_code=True
        )

    # ------------------------------------------------------------------
    # Resident modules (always kept on GPU)
    # ------------------------------------------------------------------

    def _load_resident_modules(self):
        """
        Load the tiny resident modules to GPU permanently:
          - model.embed_tokens  (~22 MB)
          - model.norm          (~5 KB)
          - model.rotary_emb    (buffers only, ~1 KB -- already materialised)

        These are NOT evicted between layers.
        """
        # embed_tokens
        embed_sd = load_file(
            str(self._shard_dir / "model.embed_tokens.safetensors"), device="cpu"
        )
        self._place_tensors(embed_sd)
        del embed_sd

        # norm
        norm_sd = load_file(
            str(self._shard_dir / "model.norm.safetensors"), device="cpu"
        )
        self._place_tensors(norm_sd)
        del norm_sd

        # rotary_emb: buffers are already materialised on CPU via AutoConfig.
        # Move them to the compute device.
        rotary = self._model.model.rotary_emb
        for buf_name, buf in list(rotary.named_buffers()):
            if buf.device.type != self.device:
                set_module_tensor_to_device(
                    self._model, f"model.rotary_emb.{buf_name}",
                    self.device, value=buf.float()
                )

        clean_memory(self.device)
        print(f"  Resident modules loaded (embed_tokens + norm + rotary_emb).")

    # ------------------------------------------------------------------
    # Per-layer shard helpers
    # ------------------------------------------------------------------

    def _place_tensors(self, state_dict: Dict[str, torch.Tensor]):
        for name, tensor in state_dict.items():
            if tensor.is_floating_point():
                set_module_tensor_to_device(
                    self._model, name, self.device, value=tensor, dtype=self.dtype
                )
            else:
                set_module_tensor_to_device(self._model, name, self.device, value=tensor)

    def _evict_layer(self, state_dict: Dict[str, torch.Tensor]):
        for name in state_dict.keys():
            set_module_tensor_to_device(self._model, name, "meta")
        # No empty_cache() -- allocator reuses freed GPU blocks automatically.

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_hooks(self):
        """
        Register pre/post hooks on every decoder layer -- same pattern
        as FluxStreamer.  Also register output hooks on target layers to
        capture hidden states.
        """
        for layer_idx in range(self._num_layers):
            layer = self._model.model.layers[layer_idx]
            layer._te_layer_idx = layer_idx

            # Load / evict hooks
            h_pre = layer.register_forward_pre_hook(self._layer_pre_hook)
            h_post = layer.register_forward_hook(self._layer_post_hook)
            self._hook_handles.extend([h_pre, h_post])

            # Capture hook on target layers
            if layer_idx in self.extract_layers:
                h_cap = layer.register_forward_hook(self._capture_hook)
                self._hook_handles.append(h_cap)

    def _layer_pre_hook(self, module: nn.Module, args):
        """Apply current layer weights; launch background GPU load for next layer."""
        idx = module._te_layer_idx
        shard_path = str(self._shard_dir / f"model.layers.{idx}.safetensors")

        # ---- Apply current layer weights --------------------------------
        if self._next_future_idx == idx and self._next_future is not None:
            # Already pre-loaded to GPU by background thread.
            gpu_sd, event = self._next_future.result()
            self._next_future = None
            self._next_future_idx = None
            torch.cuda.current_stream(self.device).wait_event(event)
            self._place_tensors(gpu_sd)
            module._te_loaded_sd = gpu_sd
        else:
            # Cold path: first layer (no pre-load yet), load synchronously.
            cpu_sd = load_file(shard_path, device="cpu")
            self._place_tensors(cpu_sd)
            module._te_loaded_sd = cpu_sd

        # ---- Launch background GPU load for NEXT layer ------------------
        next_idx = idx + 1
        if next_idx < self._num_layers and self._executor is not None:
            next_path = str(self._shard_dir / f"model.layers.{next_idx}.safetensors")
            self._next_future = self._executor.submit(
                _load_layer_to_cuda, next_path, self.device
            )
            self._next_future_idx = next_idx

    def _layer_post_hook(self, module: nn.Module, args, output):
        """Evict the layer's tensors back to meta."""
        self._evict_layer(getattr(module, "_te_loaded_sd", {}))
        module._te_loaded_sd = {}
        return output

    def _capture_hook(self, module: nn.Module, args, output):
        """Save the hidden states output from a target layer."""
        idx = module._te_layer_idx
        # Decoder layer returns a tuple; first element is hidden_states
        hidden = output[0] if isinstance(output, tuple) else output
        self._captured[idx] = hidden.detach().clone()
        return output

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        """
        Encode a single prompt and return combined hidden states.

        Returns
        -------
        prompt_embeds : torch.Tensor
            Shape (1, seq_len, 3 * hidden_dim) = (1, max_length, 7680)
            Matches joint_attention_dim of Flux2Transformer2DModel.
        """
        self._ensure_initialized()

        # Apply chat template (exactly as Flux2KleinPipeline does)
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Clear capture buffer
        self._captured.clear()

        # Run the full model.model forward (NOT model.forward to skip lm_head).
        # Hooks will load/evict each decoder layer shard automatically.
        # The model's own forward handles RoPE, causal mask, etc. correctly.
        _ = self._model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # Build the combined prompt embedding from captured layer outputs.
        # Stack: (B, num_extract, seq_len, hidden_dim)
        # Permute + reshape: (B, seq_len, num_extract * hidden_dim)
        stacked = torch.stack(
            [self._captured[k] for k in self.extract_layers], dim=1
        )
        B, num_captured, seq, hidden = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, seq, num_captured * hidden)
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)

        self._captured.clear()
        clean_memory(self.device)
        return prompt_embeds

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode pre-tokenized input_ids and return combined hidden states.

        Used by the Z-Image pipeline which handles tokenization externally
        (needs to apply Qwen3 chat template with enable_thinking=True and
        obtain the attention mask for post-hoc padding removal).

        Parameters
        ----------
        input_ids      : (B, seq_len) tensor, already on the target device
        attention_mask : (B, seq_len) bool/int tensor, or None

        Returns
        -------
        prompt_embeds : torch.Tensor
            Shape (B, seq_len, num_extract * hidden_dim)
            For Z-Image with extract_layers=(34,): (B, seq_len, 2560)
        """
        self._ensure_initialized()

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self._captured.clear()

        _ = self._model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        stacked = torch.stack(
            [self._captured[k] for k in self.extract_layers], dim=1
        )
        B, num_captured, seq, hidden = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, seq, num_captured * hidden)
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)

        self._captured.clear()
        clean_memory(self.device)
        return prompt_embeds

    @property
    def tokenizer(self):
        """Expose the tokenizer for external use (e.g. by Z-Image pipeline)."""
        self._ensure_initialized()
        return self._tokenizer

