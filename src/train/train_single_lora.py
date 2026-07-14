import os
import glob
import argparse
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
    parser = argparse.ArgumentParser(description="Train single LoRA adapter")
    parser.add_argument("--adapter_name", type=str, default="gqa_expert", help="PEFT adapter name")
    parser.add_argument("--output_dir", type=str, default="models/gqa_expert", help="Output directory for trained weights")
    parser.add_argument("--dataset_path", type=str, default="data/processed_sft_dataset", help="Path to processed dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps (-1 to disable)")
    parser.add_argument("--batch_size", type=int, default=2, help="Train batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

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

    print(f"Загрузка подготовленного датасета из: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

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
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    model = prepare_model_for_kbit_training(model)

    # Добавлены gate_proj, up_proj, down_proj
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    if hasattr(model, "add_adapter"):
        print(f"[INFO] Базовая модель уже содержит Peft-адаптеры. Добавляем новый кастомный адаптер: {args.adapter_name}...")
        model.add_adapter(peft_config, adapter_name=args.adapter_name)
        model.set_adapter(args.adapter_name)
    else:
        print(f"[INFO] Инициализируем стандартный PEFT слой для адаптера: {args.adapter_name}...")
        model = get_peft_model(model, peft_config, adapter_name=args.adapter_name)

    # Настраиваем градиенты под имя адаптера
    for name, param in model.named_parameters():
        if args.adapter_name in name:
            param.requires_grad = True
            param.data = param.data.to(torch.float32)

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

        for i in range(len(batch)):
            user_messages = [{"role": "user", "content": batch[i]["query"]}]
            user_prompt = processor.tokenizer.apply_chat_template(user_messages, tokenize=False,
                                                                  add_generation_prompt=True)

            user_inputs = processor(text=[user_prompt], images=[batch[i]["image"]], return_tensors="pt")
            user_token_len = user_inputs["input_ids"].shape[1]

            labels[i, :user_token_len] = -100

        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        remove_unused_columns=False,
        report_to="mlflow",
        run_name=f"saiga-8b-{args.adapter_name}",
        fp16=True,
        max_grad_norm=0.5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn
    )

    print(f"Запуск обучения LoRA-адаптера {args.adapter_name}...")

    with mlflow.start_run(run_name=f"saiga-8b-{args.adapter_name}"):
        mlflow.log_param("r", 8)
        mlflow.log_param("alpha", 16)
        mlflow.log_param("adapter_name", args.adapter_name)
        mlflow.log_param("dataset_size", len(dataset["train"]))

        trainer.train()

        print(f"Сохранение адаптера в {args.output_dir}...")
        trainer.model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)

    print("Обучение завершено!")


if __name__ == "__main__":
    main()