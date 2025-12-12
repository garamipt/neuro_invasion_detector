import pandas as pd
from tqdm import tqdm

df = pd.read_csv("/workspace/CIC-IDS18/03-02-2018.csv")

from df2018_to_2017 import convert_to_second_format
df_to_2017 = convert_to_second_format(df)

from dataset_transform import zeek_snort_only, del_cheater_features
df_t = zeek_snort_only(df_to_2017)
df_t = del_cheater_features(df_t)

from replace_label_marks import  convert_label_column
df_t = convert_label_column(df_t, ' Label')

import torch
import torch.nn as nn
import os

class MegaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = []
        models_names = os.listdir("/workspace/files/neuro_invasion_detector/models2")
        for model in models_names:
            try:
                self.models.append(torch.load(os.path.join( f'/workspace/files/neuro_invasion_detector/models2/{model}', 'transformer_model.pth'), weights_only=False))
            except Exception:
                pass
    
    def forward(self, x):
        output = torch.tensor([[-10000, -10000]], device=x.device)
        for model in self.models:
            out = model(x)

            if out[0][1] > output[0][1]:
                output = out

        return output
    
mega_model = MegaModel()

import pandas as pd
import numpy as np
from dataset_transform import zeek_snort_only, del_cheater_features, TrafficDataset
import joblib
import os
import torch
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from model_generator import get_model


def test_mega_model(MODEL_SAVE_DIR, df: pd.DataFrame, device='cuda'):
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

    X_test = loaded_scaler.transform(X_test)
    y_test = loaded_encoder.transform(y_test)

    NUM_CLASSES = len(np.unique(y_test))
    NUM_FEATURES = X_test.shape[1]

    print(f"Признаков: {NUM_FEATURES}, Классов: {NUM_CLASSES}")

    test_dataset = TrafficDataset(X_test, y_test, SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(test_dataset)

    f1_metric = F1Score(task="multiclass", num_classes=2, average='weighted').to(device)

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = mega_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Обновляем метрику
            f1_metric.update(predicted, labels)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Получаем итоговый F1-score
        f1 = f1_metric.compute()
        print(f'F1-score: {f1:.4f}')

    print(f"\n--- ИТОГОВАЯ ТОЧНОСТЬ НА ТЕСТЕ: {100 * correct / total:.2f}% ---")

test_mega_model(f'/workspace/files/neuro_invasion_detector/models2/{'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX'}', df_t)