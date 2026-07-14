import os
import random
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

    prompt = f"<image>\n{question}"

    return {
        "query": prompt,
        "response": answer,
        "task": task_type
    }


def main():
    instructions_path = "data/GQA-ru-train-instructions"
    images_path = "data/GQA-ru-train-images"
    output_path = "data/processed_sft_dataset"

    print("Загрузка сырых датасетов GQA-ru из папок DVC...")
    try:
        inst_dataset = load_from_disk(instructions_path)
        img_dataset = load_from_disk(images_path)
        print(f"Загружено инструкций: {len(inst_dataset)}, картинок: {len(img_dataset)}")
    except Exception as e:
        print(f"Ошибка: Не удалось загрузить датасеты. Проверь пути! {e}")
        return

    print("Индексация картинок по ID...")
    image_index = {str(img['id']): img['image'] for img in img_dataset}

    formatted_samples = []

    max_samples = min(3000, len(inst_dataset))
    random.seed(42)

    indices = list(range(len(inst_dataset)))
    random.shuffle(indices)
    selected_indices = indices[:max_samples]

    print(f"Форматирование {len(selected_indices)} примеров...")
    for idx in selected_indices:
        sample = inst_dataset[idx]
        img_id = str(sample.get('imageId'))

        raw_image = image_index.get(img_id)
        if raw_image is None:
            continue

        formatted = format_llava_sample(sample, task_type="gqa")
        formatted["image"] = raw_image
        formatted_samples.append(formatted)

    sft_dataset = Dataset.from_list(formatted_samples)

    split_dataset = sft_dataset.train_test_split(test_size=0.1, seed=42)
    final_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })

    os.makedirs(output_path, exist_ok=True)
    final_dict.save_to_disk(output_path)
    print(f"Успешно подготовлено и сохранено! Train: {len(final_dict['train'])}, Val: {len(final_dict['validation'])}")


if __name__ == "__main__":
    main()