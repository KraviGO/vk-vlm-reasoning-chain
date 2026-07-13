import os
import torch
from transformers import BitsAndBytesConfig


def get_environment_settings():
    """
    Автоматически определяет среду исполнения (Kaggle или локальный Mac).
    Конфигурирует пути к данным, параметры девайса, квантования и MLflow.
    """
    is_kaggle = os.path.exists("/kaggle/working")

    DAGSHUB_USERNAME = "YOUR_DAGSHUB_USERNAME"
    DAGSHUB_MLFLOW_URI = f"https://dagshub.com/{DAGSHUB_USERNAME}/vk-vlm-reasoning-chain.mlflow"

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
            ),
            "mlflow_tracking_uri": DAGSHUB_MLFLOW_URI
        }
    else:
        print("[ENV] Обнаружено окружение: Локальный Mac (Apple Silicon)")

        if "dagshub.com" not in DAGSHUB_MLFLOW_URI:
            for key in ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN"]:
                if key in os.environ:
                    del os.environ[key]

        return {
            "instructions_dir": os.path.join("data", "GQA-ru-test-instructions"),
            "images_dir": os.path.join("data", "GQA-ru-test-images"),
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "torch_dtype": torch.float16,
            "quantization_config": None,
            "mlflow_tracking_uri": DAGSHUB_MLFLOW_URI
        }