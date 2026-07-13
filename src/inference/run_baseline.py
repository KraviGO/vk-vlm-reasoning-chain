import os
import torch
import mlflow
from tqdm import tqdm
from datasets import load_from_disk
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer
from src.utils.env_config import get_environment_settings


def main():
    cfg = get_environment_settings()

    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
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

    model_id = "deepvk/llava-saiga-8b"
    print(f"\nЗагрузка компонентов модели {model_id}...")

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    processor.patch_size = 14
    processor.num_additional_image_tokens = 0

    model_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "torch_dtype": cfg["torch_dtype"]
    }

    if cfg["quantization_config"]:
        model_kwargs["quantization_config"] = cfg["quantization_config"]
        model_kwargs["device_map"] = "auto"

    model = LlavaForConditionalGeneration.from_pretrained(**model_kwargs)

    if cfg["device"] == "mps":
        model = model.to("mps")

    model.eval()
    print("Модель готова к инференсу!")

    is_kaggle = os.path.exists("/kaggle/working")
    eval_limit = 100 if is_kaggle else 1

    correct_predictions = 0
    evaluated_count = 0
    run_name = f"saiga-8b-baseline-{'kaggle' if is_kaggle else 'mac'}"

    print(f"\nЗапуск инференса (выборка: {eval_limit} примеров)...")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("eval_limit", eval_limit)
        mlflow.log_param("quantized_4bit", cfg["quantization_config"] is not None)

        for i in tqdm(range(eval_limit)):
            sample = test_inst[i]
            img_id = str(sample['imageId'])
            question = sample['question']
            ground_truth = sample['answer'].strip().lower()

            raw_image = image_index.get(img_id)
            if raw_image is None:
                continue

            messages = [
                {"role": "user", "content": f"<image>\n{question} Ответь одним словом."}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = processor(images=[raw_image], text=prompt, return_tensors="pt").to(cfg["device"])
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(cfg["torch_dtype"])

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][input_len:]
            predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
            predicted_answer = predicted_answer.rstrip('.')

            if predicted_answer == ground_truth:
                correct_predictions += 1

            evaluated_count += 1

        accuracy = (correct_predictions / evaluated_count) * 100 if evaluated_count > 0 else 0
        mlflow.log_metric("exact_match", accuracy)

        print(f"\n--- [ ИТОГИ БЕНЧМАРКА ] ---")
        print(f"Успешно проверено: {evaluated_count} сэмплов")
        print(f"Итоговый Exact Match: {accuracy:.2f}%")


if __name__ == "__main__":
    main()