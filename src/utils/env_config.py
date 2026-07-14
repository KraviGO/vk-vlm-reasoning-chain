import os
import torch
from transformers import BitsAndBytesConfig


def get_environment_settings():
    is_kaggle = os.path.exists("/kaggle/working")

    if is_kaggle:
        print("[ENV] Окружение: Kaggle GPU (NVIDIA T4)")
        return {
            "instructions_dir": "./data/GQA-ru-test-instructions",
            "images_dir": "./data/GQA-ru-test-images",
            "device": "cuda",
            "torch_dtype": torch.float16,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        }
    else:
        print("[ENV] Обнаружено окружение: Локальный Mac (Apple Silicon)")
        return {
            "instructions_dir": os.path.join("data", "GQA-ru-test-instructions"),
            "images_dir": os.path.join("data", "GQA-ru-test-images"),
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "torch_dtype": torch.float16,
            "quantization_config": None
        }