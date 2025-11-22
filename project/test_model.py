import pandas as pd
import numpy as np
from dataset_transform import del_cheater_features, zeek_snort_only, TrafficDataset
import joblib
import os
import torch
from torch.utils.data import DataLoader
from torchmetrics import F1Score



def test_model(MODEL_SAVE_DIR, df: pd.DataFrame, device='cuda'):
    df = del_cheater_features(df)
    df = zeek_snort_only(df)

    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values

    SEQ_LEN = 20       # Длина окна (сколько пакетов назад смотрим)
    BATCH_SIZE = 1024  # Большой батч для мощной GPU

    # 1. Загружаем сохраненные объекты
    loaded_scaler = joblib.load(os.path.join(MODEL_SAVE_DIR, 'scaler.pkl'))
    loaded_encoder = joblib.load(os.path.join(MODEL_SAVE_DIR, 'encoder.pkl'))

    # 2. Инициализируем структуру модели заново
    inference_model = torch.load(os.path.join(MODEL_SAVE_DIR, 'transformer_model.pth'), weights_only=False)
    inference_model.eval()

    print("Модель загружена для инференса.")


    X_test = loaded_scaler.transform(X_test)
    y_test = loaded_encoder.transform(y_test)

    NUM_CLASSES = len(np.unique(y_test))
    NUM_FEATURES = X_test.shape[1]

    print(f"Признаков: {NUM_FEATURES}, Классов: {NUM_CLASSES}")

    test_dataset = TrafficDataset(X_test, y_test, SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    f1_metric = F1Score(task="multiclass", num_classes=2, average='weighted').to(device)

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = inference_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Обновляем метрику
            f1_metric.update(predicted, labels)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Получаем итоговый F1-score
        f1 = f1_metric.compute()
        print(f'F1-score: {f1:.4f}')

    print(f"\n--- ИТОГОВАЯ ТОЧНОСТЬ НА ТЕСТЕ: {100 * correct / total:.2f}% ---")