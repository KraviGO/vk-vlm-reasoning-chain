import os
import argparse
import glob
import torch
import dagshub
import mlflow
from tqdm import tqdm
from datasets import load_from_disk
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer
from peft import PeftModel
from src.utils.env_config import get_environment_settings


class LLMRouter:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cpu"):
        self.device = device
        self.model_id = model_id
        print(f"\n[ROUTER] Загрузка легкой модели классификации: {model_id} на {device}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # На Mac загружаем без device_map="auto", чтобы избежать конфликтов с MPS
        if device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to("mps")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        self.model.eval()

    def route(self, question, options_text=""):
        prompt = (
            f"Тебе дан вопрос из теста. Твоя задача — классифицировать его тип.\n"
            f"Если это открытый вопрос, требующий краткого ответа словом или фразой, выведи 'GQA'.\n"
            f"Если это вопрос с выбором вариантов ответа (например, A, B, C, D), выведи 'MMBENCH'.\n\n"
            f"Вопрос: {question}\n"
            f"Варианты ответа: {options_text}\n\n"
            f"Ответ (выведи только одно слово: GQA или MMBENCH):"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=5, temperature=0.0, do_sample=False)

        gen_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
        if "MMBENCH" in gen_text:
            return "mmbench"
        return "gqa"


class HeuristicRouter:
    def route(self, sample):
        # Если в сэмпле есть непустые варианты ответов
        if any(sample.get(k) is not None and str(sample.get(k)).strip().lower() != 'nan' and str(sample.get(k)).strip() != '' for k in ['A', 'B', 'C', 'D']):
            return "mmbench"
        return "gqa"


def main():
    parser = argparse.ArgumentParser(description="Run VLM baseline inference with MoE LoRA Router")
    parser.add_argument(
        "--gqa_lora_path",
        type=str,
        default="models/gqa_expert",
        help="Path to GQA expert LoRA adapter"
    )
    parser.add_argument(
        "--mmbench_lora_path",
        type=str,
        default="models/mmbench_expert",
        help="Path to MMBench expert LoRA adapter"
    )
    parser.add_argument(
        "--router_type",
        type=str,
        default="heuristic",
        choices=["heuristic", "llm"],
        help="Router type (heuristic or llm)"
    )
    parser.add_argument(
        "--router_model_id",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID for the LLM router"
    )
    parser.add_argument(
        "--eval_limit",
        type=int,
        default=5,
        help="Number of samples to evaluate per dataset"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["gqa", "mmbench", "both"],
        help="Which task/dataset to evaluate (gqa, mmbench, or both)"
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

    eval_samples = []

    # 1. Загрузка GQA-ru тестовых данных
    if args.task in ["gqa", "both"]:
        print("\nЗагрузка тестового сплита GQA-ru...")
        try:
            test_inst = load_from_disk(cfg["instructions_dir"])['testdev']
            test_imgs = load_from_disk(cfg["images_dir"])['testdev']
            image_index = {str(img['id']): img['image'] for img in test_imgs}

            for idx in range(min(args.eval_limit, len(test_inst))):
                sample = test_inst[idx]
                eval_samples.append({
                    "source_dataset": "gqa",
                    "question": sample["question"],
                    "image": image_index.get(str(sample["imageId"])),
                    "ground_truth": sample["answer"],
                    "sample_raw": sample
                })
            print(f"Успешно добавлено вопросов GQA-ru: {min(args.eval_limit, len(test_inst))}")
        except Exception as e:
            print(f"[WARNING] Ошибка загрузки GQA-ru (проверь dvc pull): {e}")

    # 2. Загрузка MMBench-ru данных
    if args.task in ["mmbench", "both"]:
        print("\nЗагрузка сплита MMBench-ru...")
        try:
            from datasets import load_dataset
            mmbench_ds = load_dataset("deepvk/MMBench-ru")["dev"]
            for idx in range(min(args.eval_limit, len(mmbench_ds))):
                sample = mmbench_ds[idx]
                eval_samples.append({
                    "source_dataset": "mmbench",
                    "question": sample["question"],
                    "image": sample["image"],
                    "ground_truth": sample["answer"],
                    "sample_raw": sample
                })
            print(f"Успешно добавлено вопросов MMBench-ru: {min(args.eval_limit, len(mmbench_ds))}")
        except Exception as e:
            print(f"[WARNING] Ошибка загрузки MMBench-ru: {e}")

    if not eval_samples:
        print("[ERROR] Нет доступных тестовых примеров для инференса!")
        return

    # Настройки роутера
    if args.router_type == "llm":
        router = LLMRouter(model_id=args.router_model_id, device=cfg["device"])
    else:
        router = HeuristicRouter()

    # Загрузка базовой VLM-модели
    base_model_id = "deepvk/llava-saiga-8b"
    print(f"\n[INFO] Базовая модель VLM: {base_model_id}")

    print(f"\nЗагрузка процессора и токенайзера из: {base_model_id}...")
    processor = AutoProcessor.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    processor.patch_size = 14
    processor.num_additional_image_tokens = 0

    model_kwargs = {
        "pretrained_model_name_or_path": base_model_id,
        "torch_dtype": torch.float16 if cfg["device"] == "mps" else cfg["torch_dtype"]
    }

    if cfg["quantization_config"] and cfg["device"] != "mps":
        model_kwargs["quantization_config"] = cfg["quantization_config"]
        model_kwargs["device_map"] = "auto"
    elif cfg["device"] == "mps":
        model_kwargs["device_map"] = {"": "mps"}

    print("Загрузка базовой Llava модели...")
    model = LlavaForConditionalGeneration.from_pretrained(**model_kwargs)

    # Применение адаптеров через PEFT
    has_gqa_adapter = False
    has_mmbench_adapter = False

    if os.path.exists(args.gqa_lora_path):
        print(f"Применяем GQA LoRA-адаптер из {args.gqa_lora_path}...")
        model = PeftModel.from_pretrained(model, args.gqa_lora_path, adapter_name="gqa_expert")
        has_gqa_adapter = True
    else:
        print(f"[WARNING] Путь к GQA LoRA ({args.gqa_lora_path}) не найден!")

    if os.path.exists(args.mmbench_lora_path):
        if not has_gqa_adapter:
            # Если первого адаптера нет, инициализируем PEFT с MMBench
            print(f"Применяем MMBench LoRA-адаптер из {args.mmbench_lora_path}...")
            model = PeftModel.from_pretrained(model, args.mmbench_lora_path, adapter_name="mmbench_expert")
        else:
            # Иначе подгружаем второй адаптер в существующую модель
            print(f"Добавляем MMBench LoRA-адаптер из {args.mmbench_lora_path}...")
            model.load_adapter(args.mmbench_lora_path, adapter_name="mmbench_expert")
        has_mmbench_adapter = True
    else:
        print(f"[WARNING] Путь к MMBench LoRA ({args.mmbench_lora_path}) не найден!")

    if cfg["device"] == "mps" and not isinstance(model, PeftModel):
        model = model.to("mps")

    model.eval()
    print("Модель с адаптерами готова к динамическому роутингу!")

    correct_predictions = 0
    evaluated_count = 0
    run_name = f"saiga-8b-moe-router-{'mac' if cfg['device'] == 'mps' else 'gpu'}"

    print(f"\nЗапуск динамического инференса ({len(eval_samples)} примеров)...")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("base_model_id", base_model_id)
        mlflow.log_param("router_type", args.router_type)
        mlflow.log_param("gqa_adapter_loaded", has_gqa_adapter)
        mlflow.log_param("mmbench_adapter_loaded", has_mmbench_adapter)

        for sample in tqdm(eval_samples):
            question = sample["question"]
            raw_image = sample["image"]
            ground_truth = sample["ground_truth"].strip().lower()
            source = sample["source_dataset"]

            if raw_image is None:
                continue

            # 1. Роутер определяет тип задачи
            if args.router_type == "llm":
                opts = []
                for k in ["A", "B", "C", "D"]:
                    v = sample["sample_raw"].get(k)
                    if v and str(v).lower() != 'nan':
                        opts.append(f"{k}: {v}")
                opts_str = ", ".join(opts)
                predicted_task = router.route(question, opts_str)
            else:
                predicted_task = router.route(sample["sample_raw"])

            print(f"\n[ROUTER] Задача классифицирована как: {predicted_task.upper()} (Истинный источник: {source.upper()})")

            # 2. Динамическое переключение адаптера и генерация
            if predicted_task == "mmbench":
                if has_mmbench_adapter:
                    model.set_adapter("mmbench_expert")
                max_new_tokens = 1

                # Форматирование промпта с опциями множественного выбора
                prompt_parts = ["<image>"]
                hint = sample["sample_raw"].get("hint")
                if hint and str(hint).lower() != 'nan' and str(hint).strip() != '':
                    prompt_parts.append(f"Контекст: {hint}")
                prompt_parts.append(f"Вопрос: {question}")
                for k in ["A", "B", "C", "D"]:
                    v = sample["sample_raw"].get(k)
                    if v and str(v).lower() != 'nan' and str(v).strip() != '':
                        prompt_parts.append(f"{k}. {v}")
                prompt_parts.append("Ответ:")
                prompt_text = "\n".join(prompt_parts)
            else:
                if has_gqa_adapter:
                    model.set_adapter("gqa_expert")
                max_new_tokens = 3
                prompt_text = f"<image>\n{question}\nОтветь одним словом."

            # Формирование промпта в Saiga-шаблоне
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = processor(images=[raw_image], text=prompt, return_tensors="pt").to(cfg["device"])
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16 if cfg["device"] == "mps" else cfg["torch_dtype"])

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][input_len:]
            predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()

            print(f"[Q]: {question}")
            print(f"[Prompt]: {prompt_text}")
            print(f"[A Model]: {predicted_answer}")
            print(f"[GT]: {ground_truth}")
            print("-" * 30)

            # Проверка правильности предсказания
            if predicted_task == "mmbench":
                # Для MMBench ищем точное совпадение с буквой (в нижнем регистре)
                is_correct = (predicted_answer == ground_truth)
            else:
                is_correct = (ground_truth in predicted_answer)

            if is_correct:
                correct_predictions += 1
            evaluated_count += 1

        accuracy = (correct_predictions / evaluated_count) * 100 if evaluated_count > 0 else 0
        mlflow.log_metric("accuracy", accuracy)

        print(f"\n--- [ ИТОГИ РАБОТЫ РОУТЕРА ] ---")
        print(f"Всего проверено: {evaluated_count} примеров")
        print(f"Точность (Accuracy): {accuracy:.2f}%")


if __name__ == "__main__":
    main()