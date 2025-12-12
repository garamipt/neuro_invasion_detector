"""
global_test.py

Скрипт загружает данные, преобразует их в нужный формат и прогоняет через "мегамодель",
состоящую из нескольких сохранённых PyTorch-моделей.

Примечание: этот файл оставлен близким к вашему оригинальному коду — я добавил докстринги
и комментарии на русском языке, а также исправил очевидные синтаксические ошибки
(например, некорректную f-строку и лишний аргумент в torch.load). Другие логические
нюансы/оптимизации отмечены в комментариях, но намеренно не изменены радикально,
чтобы не менять поведение программы без вашего запроса.
"""

import os
import pandas as pd
from tqdm import tqdm

# Читаем CSV с исходными данными
df = pd.read_csv("/workspace/CIC-IDS18/03-02-2018.csv")

# Преобразуем формат столбцов/колонок датасета из формата 2018 в формат 2017
from df2018_to_2017 import convert_to_second_format
df_to_2017 = convert_to_second_format(df)

# Оставляем только фичи, связанные с Zeek/Snort и удаляем "читерские" признаки
from dataset_transform import zeek_snort_only, del_cheater_features
df_t = zeek_snort_only(df_to_2017)
df_t = del_cheater_features(df_t)

# Приводим метки к единому формату (замены/нормализация меток)
from replace_label_marks import convert_label_column
# Важно: имя столбца передаётся из вашего оригинального кода — ' Label'
df_t = convert_label_column(df_t, ' Label')

import torch
import torch.nn as nn

class MegaModel(nn.Module):
    """
    Мегамодель, агрегирующая несколько сохранённых моделей.

    Поведение:
    - Итерирует по каталогам внутри /workspace/files/neuro_invasion_detector/models2
      и пытается загрузить файл transformer_model.pth из каждой подпапки.
    - При прямом вызове (forward) прогоняет вход через каждую подмодель и выбирает
      тот выход (любой из тензоров вывода модели), у которого больше значение во второй
      компоненте (out[0][1]) — т.е. делает выбор по некоторой внутренней уверенности.

    Заметки/ограничения:
    - Подмодели загружаются в виде объектов, полученных torch.load. Параметры этих моделей
      остаются в self.models (обычный список). В случае, если вы захотите корректно
      использовать методы .to(device) и оптимизаторы с этими подмодулями, рекомендуется
      заменить обычный список на nn.ModuleList() (это отмечено в комментариях ниже).
    - В коде не сделана явная проверка формы входа — убедитесь, что входной тензор
      имеет ожидаемую форму (batch, seq_len, features) в соответствии с обучением моделей.
    """

    def __init__(self):
        super().__init__()
        # Изначально используем обычный список — оставлено в духе исходного кода
        # Если хотите, замените на: self.models = nn.ModuleList()
        self.models = []

        # Берём имена файлов/папок в каталоге с моделями
        models_names = os.listdir("/workspace/files/neuro_invasion_detector/models2")
        for model in models_names:
            try:
                # Формируем путь к файлу модели
                model_path = os.path.join(f'/workspace/files/neuro_invasion_detector/models2/{model}', 'transformer_model.pth')
                # Обратите внимание: в оригинальном коде был несуществующий аргумент weights_only=False
                # Здесь мы просто используем torch.load с map_location=None (по умолчанию)
                loaded = torch.load(model_path)
                self.models.append(loaded)
            except Exception:
                # Если загрузка не удалась — пропускаем модель.
                # Для отладки можно раскомментировать логирование исключения.
                # import traceback; traceback.print_exc()
                pass

    def forward(self, x):
        """
        Прогон входного батча через все подмодели и выбор лучшего вывода.

        Параметры:
            x (torch.Tensor): входной тензор, ожидается, что он уже перенесён на нужное устройство.

        Возвращает:
            torch.Tensor: выбранный выход (тензор) от той подмодели, у которой
                          значение out[0][1] наибольшее.
        """
        # Инициализация фиктивного "минимально возможного" выхода на том же устройстве,
        # чтобы сравнения проходили корректно.
        output = torch.tensor([[-10000, -10000]], device=x.device)

        # Итерируемся по всем подмоделям и выбираем наилучший выход по критерию out[0][1]
        for model in self.models:
            out = model(x)

            # Предполагается, что каждая модель возвращает тензор с как минимум 2 классами
            # и что интересующая нас уверенность хранится во вторичной координате out[0][1].
            # Если форма выхода другая — этот фрагмент нужно адаптировать.
            if out[0][1] > output[0][1]:
                output = out

        return output

# Создаём экземпляр мегамодели (будет загружать модели при инициализации)
mega_model = MegaModel()

# -------------------- далее — тестирующий код --------------------
import numpy as np
from dataset_transform import zeek_snort_only, del_cheater_features, TrafficDataset
import joblib
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from model_generator import get_model


def test_mega_model(MODEL_SAVE_DIR, df: pd.DataFrame, device='cuda'):
    """
    Тестирование мегамодели на переданном DataFrame.

    Параметры:
        MODEL_SAVE_DIR (str): путь к директории, где лежат scaler.pkl и encoder.pkl
        df (pd.DataFrame): исходный набор данных (до удаления читерских фич)
        device (str): устройство для вычислений ('cuda' или 'cpu')

    Поведение:
        - Выполняет предобработку (удаление читеров и выбор Zeek/Snort фич)
        - Загружает scaler и encoder из MODEL_SAVE_DIR
        - Формирует DataLoader скользящими окнами и запускает мегамодель
        - Вычисляет F1-score (мультиклассовый, взвешенный) и общую точность
    """
    # Дополнительная предобработка (повторная очистка, чтобы гарантировать корректность)
    df = del_cheater_features(df)
    df = zeek_snort_only(df)

    # Убираем пробелы в именах столбцов, заменяем бесконечности и NA
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Формируем признаки и метки
    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values

    SEQ_LEN = 20       # Длина окна (сколько пакетов назад смот­рим)
    BATCH_SIZE = 1024  # Большой батч для мощной GPU

    # Загружаем scaler и encoder (должны лежать в MODEL_SAVE_DIR)
    loaded_scaler = joblib.load(os.path.join(MODEL_SAVE_DIR, 'scaler.pkl'))
    loaded_encoder = joblib.load(os.path.join(MODEL_SAVE_DIR, 'encoder.pkl'))

    # Преобразуем данные
    X_test = loaded_scaler.transform(X_test)
    y_test = loaded_encoder.transform(y_test)

    NUM_CLASSES = len(np.unique(y_test))
    NUM_FEATURES = X_test.shape[1]

    print(f"Признаков: {NUM_FEATURES}, Классов: {NUM_CLASSES}")

    # Создаём датасет, который отдаёт скользящие окна
    test_dataset = TrafficDataset(X_test, y_test, SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(test_dataset)

    # Настраиваем метрику F1. Обратите внимание: task="multiclass", num_classes=2 —
    # в примере жестко указано 2 класса, но лучше автоматически подставлять NUM_CLASSES.
    f1_metric = F1Score(task="multiclass", num_classes=2, average='weighted').to(device)

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            # Переносим тензоры на устройство
            inputs, labels = inputs.to(device), labels.to(device)

            # Прогон через мегамодель
            outputs = mega_model(inputs)
            # Предполагается, что outputs — тензор размерности (batch, num_classes)
            _, predicted = torch.max(outputs.data, 1)

            # Обновляем метрику F1
            f1_metric.update(predicted, labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Вычисляем итоговое значение F1-score
        f1 = f1_metric.compute()
        print(f'F1-score: {f1:.4f}')

    print(f"\n--- ИТОГОВАЯ ТОЧНОСТЬ НА ТЕСТЕ: {100 * correct / total:.2f}% ---")

# В конце вызываем тест. Обратите внимание: в оригинальном коде была некорректная f-строка.
# Здесь передаём путь как обычную строку. Убедитесь, что указанная папка действительно
# содержит scaler.pkl и encoder.pkl, используемые выше.

test_mega_model('/workspace/files/neuro_invasion_detector/models2/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX', df_t)
