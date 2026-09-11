from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, PeftModel
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


@dataclass
class QLoRASettings:
    model_id: str
    use_4bit: bool = False
    use_bf16: bool = False
    trust_remote_code: bool = True  # for models that require custom code (Qwen-VL-Chat), but fine for Qwen2


def load_quantized_vlm(
    settings: QLoRASettings,
    trainable: bool = True,
    adapter_path: Optional[Union[str, Path]] = None,
):
    with tqdm(total=4, desc="Loading VLM", unit="step") as pbar:
        processor = AutoProcessor.from_pretrained(
            adapter_path if adapter_path and Path(adapter_path).exists() else settings.model_id,
            trust_remote_code=settings.trust_remote_code,
        )
        
        # EXPERIMENT A: Enable Dynamic Resolution up to 1024x1024 pixels (1024 tokens)
        processor.image_processor.min_pixels = 256 * 28 * 28
        processor.image_processor.max_pixels = 1024 * 28 * 28
        pbar.update(1)

        compute_dtype = torch.bfloat16 if settings.use_bf16 else torch.float16

        bnb_config = None
        if settings.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )

        device_map = {"": 0} if trainable else "auto"

        pbar.set_description("Loading model weights")
        model = AutoModelForImageTextToText.from_pretrained(
            settings.model_id,
            quantization_config=bnb_config,
            dtype=compute_dtype,
            device_map=device_map,
            trust_remote_code=settings.trust_remote_code,
        )
        pbar.update(1)

        if adapter_path and Path(adapter_path).exists():
            pbar.set_description("Loading LoRA adapter")
            model = PeftModel.from_pretrained(model, adapter_path)
            if not trainable:
                model.eval()
            pbar.update(2)
            return model, processor, compute_dtype

        if trainable:
            pbar.set_description("Preparing model for training")
            if settings.use_4bit:
                model = prepare_model_for_kbit_training(model)

            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            pbar.update(2)
        else:
            pbar.update(2)

    return model, processor, compute_dtype
