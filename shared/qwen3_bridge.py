"""Small compatibility facade around TransformerLens' native Qwen3-VL bridge."""

from __future__ import annotations

import os
from typing import Any

import torch
from PIL import Image


class Qwen3VLBridgeAdapter:
    """Expose the legacy experiment call shape on top of ``TransformerBridge``."""

    def __init__(self, bridge, model_name: str):
        self.bridge = bridge
        self.cfg = bridge.cfg
        self.processor = bridge.processor
        self.model_name = model_name
        self.hf_model = bridge.original_model
        self.vision_model = self.hf_model.model.visual

    @staticmethod
    def _normalize_images(text, images):
        if images is None:
            return None
        if isinstance(text, str):
            if isinstance(images, (list, tuple)) and len(images) == 1:
                first = images[0]
                if isinstance(first, Image.Image):
                    return first
                if isinstance(first, (list, tuple)) and len(first) == 1:
                    return first[0]
            return images
        if isinstance(images, (list, tuple)):
            return [item[0] if isinstance(item, (list, tuple)) and len(item) == 1 else item
                    for item in images]
        return images

    def _prepare(self, input, images):
        if images is None or isinstance(input, torch.Tensor):
            return input, {}
        prepared = self.bridge.prepare_multimodal_inputs(
            input, self._normalize_images(input, images)
        )
        tokens = prepared.pop("input_ids")
        return tokens, prepared

    def __call__(self, input, images=None, **kwargs):
        tokens, multimodal = self._prepare(input, images)
        multimodal.update(kwargs)
        return self.bridge(tokens, **multimodal)

    def run_with_cache(self, input, images=None, **kwargs):
        tokens, multimodal = self._prepare(input, images)
        multimodal.update(kwargs)
        return self.bridge.run_with_cache(tokens, **multimodal)

    def run_with_hooks(self, input, images=None, **kwargs):
        tokens, multimodal = self._prepare(input, images)
        multimodal.update(kwargs)
        return self.bridge.run_with_hooks(tokens, **multimodal)

    def to_tokens(
        self,
        input,
        images=None,
        *,
        return_also_image_values: bool = False,
        **kwargs,
    ):
        if images is None:
            return self.bridge.to_tokens(input, **kwargs)
        tokens, multimodal = self._prepare(input, images)
        if not return_also_image_values:
            return tokens
        return tokens, multimodal

    def to_string(self, *args, **kwargs):
        return self.bridge.to_string(*args, **kwargs)

    def eval(self):
        self.bridge.eval()
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self.bridge, name)


def load_qwen3_bridge(
    model_name: str,
    *,
    device: str | torch.device,
    device_list: list[int] | None,
    torch_dtype: torch.dtype,
    cache_dir: str | None = None,
):
    if cache_dir is not None:
        # TransformerBridge does not expose from_pretrained(cache_dir=...), but
        # it imports Hugging Face lazily, so this standard cache override has
        # identical effect for this one-model-per-process experiment setup.
        os.environ["HF_HUB_CACHE"] = cache_dir

    from transformer_lens.model_bridge import TransformerBridge

    load_kwargs: dict[str, Any] = {
        "model_name": model_name,
        "dtype": torch_dtype,
    }
    if device_list:
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "--device-list must be set before CUDA is initialized for Qwen3-VL"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_list))
        load_kwargs["n_devices"] = len(device_list)
    else:
        load_kwargs["device"] = device

    bridge = TransformerBridge.boot_transformers(**load_kwargs)
    bridge.eval()
    return Qwen3VLBridgeAdapter(bridge, model_name), bridge.processor
