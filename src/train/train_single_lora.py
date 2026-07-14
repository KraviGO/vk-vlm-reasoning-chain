import os
import glob
import torch
import dagshub
import mlflow
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk


def main():
    try:
        dagshub.init(
            repo_owner="KraviGO",
            repo_name="vk-vlm-reasoning-chain",
            mlflow=True
        )

        experiment_name = "vk-vlm-reasoning-chain"
        mlflow.set_experiment(experiment_name)
        print("[INFO] MLflow успешно инициализирован!")
    except Exception as e:
        print(f"[WARNING] Ошибка инициализации MLflow/DagsHub: {e}. Логирование будет локальным.")

    kaggle_configs = glob.glob("/kaggle/input/**/config.json", recursive=True)
    if kaggle_configs:
        model_id = os.path.dirname(kaggle_configs[0])
        print(f"\n[INFO] Найдена локальная модель в Kaggle: {model_id}")
    else:
        model_id = "deepvk/llava-saiga-8b"
        print(f"\n[INFO] Локальная модель не найдена. Обучаем на базе HF: {model_id}")

    dataset_path = "data/processed_sft_dataset"
    output_dir = "models/single_lora_baseline"

    print("Загрузка подготовленного датасета...")
    dataset = load_from_disk(dataset_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Загрузка процессора и квантованной модели...")
    processor = AutoProcessor.from_pretrained(model_id)

    processor.patch_size = 14
    processor.num_additional_image_tokens = 0

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    if hasattr(model, "add_adapter"):
        print("[INFO] Базовая модель уже содержит Peft-адаптеры. Добавляем новый кастомный адаптер...")
        model.add_adapter(peft_config, adapter_name="single_lora_baseline")
        model.set_adapter("single_lora_baseline")
    else:
        print("[INFO] Инициализируем стандартный PEFT слой...")
        model = get_peft_model(model, peft_config)

    for name, param in model.named_parameters():
        if "single_lora_baseline" in name:
            param.requires_grad = True

    def collate_fn(batch):
        texts = []
        images = []
        for example in batch:
            messages = [
                {"role": "user", "content": example["query"]},
                {"role": "assistant", "content": example["response"]}
            ]
            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(prompt)
            images.append(example["image"])

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = inputs["input_ids"].clone()

        for i in range(len(texts)):
            user_messages = [{"role": "user", "content": batch[i]["query"]}]
            user_prompt = processor.tokenizer.apply_chat_template(user_messages, tokenize=False,
                                                                  add_generation_prompt=True)
            user_token_len = len(processor.tokenizer.encode(user_prompt))

            labels[i, :user_token_len] = -100

        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        remove_unused_columns=False,
        report_to="mlflow",
        run_name="saiga-8b-single-lora"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn
    )

    print("Запуск обучения Single-LoRA...")

    with mlflow.start_run(run_name="saiga-8b-single-lora"):
        mlflow.log_param("r", 8)
        mlflow.log_param("alpha", 16)
        mlflow.log_param("dataset_size", len(dataset["train"]))

        trainer.train()

        print("Сохранение адаптера...")
        trainer.model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    print("Обучение завершено!")


if __name__ == "__main__":
    main()