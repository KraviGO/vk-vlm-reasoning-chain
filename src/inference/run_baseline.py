import os
import argparse
import glob
import torch
import dagshub
import mlflow
from tqdm import tqdm
from datasets import load_from_disk
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer
from peft import PeftModel  # <--- Добавили импорт PEFT
from src.utils.env_config import get_environment_settings


def main():
    parser = argparse.ArgumentParser(description="Run VLM baseline inference with LoRA")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/single_lora_baseline",  # <--- По дефолту смотрим на наш скачанный адаптер
        help="Path to local LoRA weights or base model ID"
    )
    args = parser.parse_args()

    cfg = get_environment_settings()

    dagshub.init(
        repo_owner="KraviGO",
        repo_name="vk-vlm-reasoning-chain",
        mlflow=True
    )

    experiment_name = "vk-vlm-reasoning-chain"
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is not None and exp.lifecycle_stage == "deleted":
            print(f"Эксперимент '{experiment_name}' в корзине. Создаем v2...")
            experiment_name = f"{experiment_name}-v2"
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Ошибка MLflow: {e}. Используем резервный vlm-fallback...")
        mlflow.set_experiment("vlm-fallback-experiment")

    print("\nЗагрузка тестового сплита с диска...")
    try:
        test_inst = load_from_disk(cfg["instructions_dir"])['testdev']
        test_imgs = load_from_disk(cfg["images_dir"])['testdev']
        print(f"Успешно загружено вопросов: {len(test_inst)}, картинок: {len(test_imgs)}")
    except Exception as e:
        print(f"Ошибка загрузки данных (проверь dvc pull): {e}")
        return

    print("Индексация изображений по ID...")
    image_index = {str(img['id']): img['image'] for img in test_imgs}

    # Жестко прописываем ID базовой модели, на которой учился адаптер
    base_model_id = "deepvk/llava-saiga-8b"
    lora_path = args.model_path

    print(f"\n[INFO] Базовая модель: {base_model_id}")
    print(f"[INFO] Путь к LoRA-адаптеру: {lora_path}")

    print(f"\nЗагрузка компонентов процессора из: {base_model_id}...")
    processor = AutoProcessor.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    processor.patch_size = 14
    processor.num_additional_image_tokens = 0

    # Настройки загрузки базовой модели
    model_kwargs = {
        "pretrained_model_name_or_path": base_model_id,
        "torch_dtype": torch.float16 if cfg["device"] == "mps" else cfg["torch_dtype"]
    }

    # Квантование работает в Кагле (Linux/CUDA), на Маке (MPS) пропускаем его
    if cfg["quantization_config"] and cfg["device"] != "mps":
        model_kwargs["quantization_config"] = cfg["quantization_config"]
        model_kwargs["device_map"] = "auto"
    elif cfg["device"] == "mps":
        # Для Мака явная карта девайсов, чтобы не ругался accelerate
        model_kwargs["device_map"] = {"": "mps"}

    print("Загрузка базовой LLovA модели...")
    model = LlavaForConditionalGeneration.from_pretrained(**model_kwargs)

    # Накатываем LoRA веса
    if os.path.exists(lora_path):
        print(f"Применяем LoRA-адаптер из {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        print(f"[WARNING] Путь {lora_path} не найден! Запуск чистой базовой модели.")

    if cfg["device"] == "mps" and not isinstance(model, PeftModel):
        model = model.to("mps")

    model.eval()
    print("Модель с LoRA адаптером готова к инференсу!")

    is_kaggle = os.path.exists("/kaggle/working")
    # Если мы на Маке, давай уменьшим лимит до 5-10 штук для быстрой проверки, чтобы не ждать часами
    eval_limit = 100 if is_kaggle else 5

    correct_predictions = 0
    evaluated_count = 0
    run_name = f"saiga-8b-lora-{'kaggle' if is_kaggle else 'mac'}"

    print(f"\nЗапуск инференса (выборка: {eval_limit} примеров)...")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("base_model_id", base_model_id)
        mlflow.log_param("lora_path", lora_path)
        mlflow.log_param("eval_limit", eval_limit)

        for i in tqdm(range(eval_limit)):
            sample = test_inst[i]
            img_id = str(sample['imageId'])
            question = sample['question']
            ground_truth = sample['answer'].strip().lower()

            raw_image = image_index.get(img_id)
            if raw_image is None:
                continue

            # Формируем промпт под Saiga-шаблон рассуждений
            messages = [
                {"role": "user", "content": f"<image>\n{question}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = processor(images=[raw_image], text=prompt, return_tensors="pt").to(cfg["device"])
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16 if cfg["device"] == "mps" else cfg["torch_dtype"])

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,  # <--- Увеличил, чтобы увидеть reasoning chain (а не 1 слово)
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][input_len:]
            predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()

            print(f"\n[Q]: {question}")
            print(f"[A Model]: {predicted_answer}")
            print(f"[GT]: {ground_truth}\n" + "-"*20)

            # Простая эвристика проверки (находится ли таргет-слово в ответе)
            if ground_truth in predicted_answer:
                correct_predictions += 1

            evaluated_count += 1

        accuracy = (correct_predictions / evaluated_count) * 100 if evaluated_count > 0 else 0
        mlflow.log_metric("exact_match", accuracy)

        print(f"\n--- [ ИТОГИ БЕНЧМАРКА LO RA ] ---")
        print(f"Успешно проверено: {evaluated_count} сэмплов")
        print(f"Итоговый Exact Match: {accuracy:.2f}%")


if __name__ == "__main__":
    main()