import gc
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def load_processor(model_id, min_pixels, max_pixels):
    processor = AutoProcessor.from_pretrained(model_id)
    processor.image_processor.min_pixels = int(min_pixels)
    processor.image_processor.max_pixels = int(max_pixels)
    return processor


def load_base_model(model_id, device):
    dtype = torch.float16 if device == "cuda" else torch.float32
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto"
    )


def build_lora_model(model, cfg):
    lora_config = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "merger.mlp",
            "attn.qkv", "attn.proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def load_adapter_for_eval(model_id, adapter_path, device):
    processor = AutoProcessor.from_pretrained(adapter_path)
    base_model = load_base_model(model_id, device)
    model = PeftModel.from_pretrained(base_model, adapter_path).eval()
    return processor, model


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
