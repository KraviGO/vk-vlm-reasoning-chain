import os
import random
import argparse
from datasets import load_from_disk, Dataset, DatasetDict


def format_llava_sample(sample, task_type):
    """
    Приводит GQA-ru к единому формату для SFTTrainer.
    Каждый пример должен содержать:
    - image: PIL.Image
    - query: текст промпта с тегом <image>
    - response: целевой ответ
    """
    question = sample.get("question", "")
    answer = sample.get("answer", "")

    if task_type == "gqa":
        prompt = f"<image>\n{question}\nОтветь одним словом."
    else:
        prompt = f"<image>\n{question}"

    return {
        "query": prompt,
        "response": answer,
        "task": task_type
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for SFT Trainer")
    parser.add_argument("--task", type=str, default="gqa", choices=["gqa", "mmbench"], help="Task type (gqa or mmbench)")
    parser.add_argument("--instructions_path", type=str, default="data/GQA-ru-train-instructions", help="Path to instructions dataset")
    parser.add_argument("--images_path", type=str, default="data/GQA-ru-train-images", help="Path to images dataset")
    parser.add_argument("--output_path", type=str, default="data/processed_sft_dataset", help="Output path for processed dataset")
    parser.add_argument("--max_samples", type=int, default=3000, help="Maximum number of samples to process")
    args = parser.parse_args()

    print(f"Запуск подготовки данных для задачи: {args.task}")
    print("Загрузка сырых датасетов из папок DVC...")
    try:
        inst_data = load_from_disk(args.instructions_path)
        img_data = load_from_disk(args.images_path)

        if isinstance(inst_data, DatasetDict):
            split_name = list(inst_data.keys())[0]
            print(f"[INFO] Инструкции загружены как DatasetDict. Используем сплит: {split_name}")
            inst_dataset = inst_data[split_name]
        else:
            inst_dataset = inst_data

        if isinstance(img_data, DatasetDict):
            split_name = list(img_data.keys())[0]
            print(f"[INFO] Картинки загружены как DatasetDict. Используем сплит: {split_name}")
            img_dataset = img_data[split_name]
        else:
            img_dataset = img_data

        print(f"Загружено инструкций: {len(inst_dataset)}, картинок: {len(img_dataset)}")
    except Exception as e:
        print(f"Ошибка: Не удалось загрузить датасеты. Проверь пути! {e}")
        return

    print("Индексация картинок по ID...")
    image_index = {}
    for i in range(len(img_dataset)):
        row = img_dataset[i]
        img_id = str(row.get('id', row.get('imageId', '')))
        img_val = row.get('image')
        if img_id and img_val:
            image_index[img_id] = img_val

    print(f"Индексировано уникальных картинок: {len(image_index)}")

    formatted_samples = []

    max_samples = min(args.max_samples, len(inst_dataset))
    random.seed(42)

    indices = list(range(len(inst_dataset)))
    random.shuffle(indices)
    selected_indices = indices[:max_samples]

    print(f"Форматирование {len(selected_indices)} примеров...")
    for idx in selected_indices:
        sample = inst_dataset[idx]
        img_id = str(sample.get('imageId', sample.get('id', '')))

        raw_image = image_index.get(img_id)
        if raw_image is None:
            continue

        formatted = format_llava_sample(sample, task_type=args.task)
        formatted["image"] = raw_image
        formatted_samples.append(formatted)

    if not formatted_samples:
        print("[ERROR] Не удалось сопоставить инструкции с картинками! Проверь ключи ID.")
        return

    sft_dataset = Dataset.from_list(formatted_samples)

    split_dataset = sft_dataset.train_test_split(test_size=0.1, seed=42)
    final_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })

    os.makedirs(args.output_path, exist_ok=True)
    final_dict.save_to_disk(args.output_path)
    print(f"Успешно подготовлено и сохранено! Train: {len(final_dict['train'])}, Val: {len(final_dict['validation'])}")


if __name__ == "__main__":
    main()