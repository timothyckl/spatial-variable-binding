"""Raw-HF Qwen3-VL helpers for the COCO amplification experiment."""

from __future__ import annotations

import gc

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from . import common as cc
from .qwen import QWEN_IMAGE_PAD_TOKEN, QwenCocoRunner, resolve_lm_layers


MODEL_IDS = {
    "qwen3-vl-8b-instruct": "Qwen/Qwen3-VL-8B-Instruct",
}
QWEN3_PATCH_SIZE = 16
QWEN3_FACTOR = 32


def canonical_model_name(name: str) -> str:
    key = name.lower()
    aliases = {
        "qwen3vl": "qwen3-vl-8b-instruct",
        "qwen3-vl-8b": "qwen3-vl-8b-instruct",
        "qwen3-vl-8b-instruct": "qwen3-vl-8b-instruct",
        "qwen/qwen3-vl-8b-instruct": "qwen3-vl-8b-instruct",
    }
    if key not in aliases:
        raise ValueError(f"unsupported COCO amplification model {name!r}")
    return aliases[key]


class Qwen3CocoRunner(QwenCocoRunner):
    """Qwen3 variant of the established raw-HF Qwen COCO runner.

    Probe features come from the final vision merger (``pooler_output``), and
    amplification is applied to image positions at the first LM block input.
    """

    def __init__(self, *, model_name: str, dtype_name: str, img_dir: str,
                 cache_dir: str | None = None):
        self.model_slug = canonical_model_name(model_name)
        self.model_id = MODEL_IDS[self.model_slug]
        self.dtype_name = dtype_name
        self.torch_dtype = cc.torch_dtype_from_name(dtype_name)
        self.img_dir = img_dir

        print(f"Loading {self.model_id} ...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            cache_dir=cache_dir,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.image_pad_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            QWEN_IMAGE_PAD_TOKEN
        )
        assert (
            self.image_pad_token_id is not None
            and self.image_pad_token_id != self.processor.tokenizer.unk_token_id
        ), f"tokenizer does not know {QWEN_IMAGE_PAD_TOKEN}"

        self.model_device = next(self.model.parameters()).device
        self.vision_model = self.model.model.visual
        self.language_model, self.lm_layers = resolve_lm_layers(self.model)
        assert getattr(self.model.config.vision_config, "spatial_merge_size", 2) == 2

        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        print(f"d_model = {text_cfg.hidden_size} | n_layers = {text_cfg.num_hidden_layers}")
        print(
            "visual spatial_merge_size = "
            f"{getattr(cfg.vision_config, 'spatial_merge_size', 'unknown')}"
        )
        print(
            f"MODEL_DTYPE = {dtype_name} | actual parameter dtype = "
            f"{next(self.model.parameters()).dtype}"
        )
        print(f"  Visual module:   {type(self.vision_model).__name__}")
        print(f"  Language module: {type(self.language_model).__name__}")
        print(f"  LM layers:       {len(self.lm_layers)}")

    def bbox_to_patch_coords_qwen2(self, box, img_size, grid_h=None, grid_w=None,
                                   patch_size=QWEN3_PATCH_SIZE, factor=QWEN3_FACTOR):
        return super().bbox_to_patch_coords_qwen2(
            box,
            img_size,
            grid_h=grid_h,
            grid_w=grid_w,
            patch_size=patch_size,
            factor=factor,
        )
