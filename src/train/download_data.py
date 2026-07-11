import os
from datasets import load_dataset
from huggingface_hub import login

from datasets import logging
logging.set_verbosity_info()


def main():
    print("Авторизация в Hugging Face...")

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("Переменная окружения HF_TOKEN не найдена.")
    else:
        login(token=hf_token, add_to_git_credential=False)
        print("Успешно авторизовались через HF_TOKEN!")

        configs = {
            "train_balanced_instructions": "GQA-ru-train-instructions",
            "train_balanced_images": "GQA-ru-train-images",
            "testdev_balanced_instructions": "GQA-ru-test-instructions",
            "testdev_balanced_images": "GQA-ru-test-images"
        }

    for config_name, folder_name in configs.items():
        print(f"\nСтарт загрузки конфигурации '{config_name}' deepvk/GQA-ru...")

        output_dir = os.path.join("data", folder_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            dataset = load_dataset("deepvk/GQA-ru", config_name)

            print(f"Конфигурация '{config_name}' успешно загружена!")
            print(f"Доступные сплиты: {list(dataset.keys())}")

            dataset.save_to_disk(output_dir)
            print(f"Сохранено в локальную директорию: {output_dir}")

            active_split = list(dataset.keys())[0]
            sample = dataset[active_split][0]

            print(f"Пример структуры данных ({active_split} split, строка 0):")
            for key, value in sample.items():
                if key == 'image' or hasattr(value, 'save'):
                    print(f"  {key}: {type(value)} (Image)")
                else:
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"Произошла ошибка при загрузке '{config_name}': {e}")

    print("\nВсе компоненты датасета GQA-ru успешно скачаны!")


if __name__ == "__main__":
    main()