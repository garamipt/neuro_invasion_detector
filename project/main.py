"""
main.py

Главный скрипт для подготовки данных, обучения и тестирования моделей.

Сценарий работы:
    1. Проходит по CSV-файлам в директории READ_DIR.
    2. Для каждого файла (за исключением списка skip): создаёт папку для сохранения
       результатов, делит данные на train/test, формирует DataLoader, создаёт модель,
       тренирует её и сохраняет на диск, затем прогоняет тестирование.

Важно:
    - Ожидается, что в проекте присутствуют модули:
        dataset_transform.get_dataset
        model_generator.get_model
        learning.get_trainde_model
        test_model.test_model
    - Параметры путей READ_DIR и SAVE_DIR можно менять в начале файла.
"""

from dataset_transform import get_dataset
from model_generator import get_model
from learning import get_trainde_model
from test_model import test_model
import os
import pandas as pd
import torch

# Директории для чтения исходных CSV и сохранения обученных моделей
READ_DIR = './files'
SAVE_DIR = './models'

# Список файлов, которые хотим пропустить (например, слишком большие или уже обработанные)
skip = [
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Monday-WorkingHours.pcap_ISCX.csv'
]

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")

def create_folder_if_not_exists(folder_path):
    """
    Создаёт папку, если она не существует.

    Параметры:
        folder_path (str): путь к папке, которую нужно создать.

    Возвращает:
        None

    Поведение:
        - Если папка уже существует, выводит сообщение об этом.
        - Если папки нет — создаёт её рекурсивно и выводит сообщение.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Папка создана: {folder_path}")
    else:
        print(f"Папка уже существует: {folder_path}")


# Основной цикл: обходим все файлы в READ_DIR
for file_name in os.listdir(READ_DIR):
    # Небольшая отладочная печать — какой файл рассматриваем и стоит ли он в skip
    print(file_name, file_name in skip)

    # Пропускаем файлы из списка skip
    if file_name in skip:
        print(file_name, 'skiped')
        continue

    # Создаём папку для сохранения результатов конкретного файла (название без .csv)
    target_folder = f"{SAVE_DIR}/{file_name[:-4]}"
    create_folder_if_not_exists(target_folder)

    # Читаем CSV в DataFrame
    df = pd.read_csv(f"{READ_DIR}/{file_name}")

    # Разделяем на тренировочную и тестовую части (например, 90% / 10%)
    train_df = df[:int(len(df) * 0.9)]
    test_df = df[int(len(df) * 0.9):]

    # Формируем DataLoader и получаем служебную информацию INFO
    # get_dataset должен возвращать (train_loader, INFO), где INFO используется для создания модели
    train_loader, INFO = get_dataset(train_df, target_folder)

    # Создаём модель, используя INFO (например NUM_FEATURES, NUM_CLASSES)
    model = get_model(*INFO, device=device)

    # Тренируем модель и получаем обученный экземпляр
    # get_trainde_model принимает модель, train_loader и число эпох (в примере 15)
    model = get_trainde_model(model, train_loader, 15, device=device)

    # Сохраняем модель на диск в формате PyTorch (возможна замена на state_dict при желании)
    torch.save(model, f"{target_folder}/transformer_model.pth")
    print("Transformer модель сохранена.")

    # Запускаем тестирование модели на отложенной выборке
    test_model(target_folder, test_df, device=device)
