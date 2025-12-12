import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import numpy as np
from dataset_transform import zeek_snort_only, del_cheater_features
import joblib
from torch.utils.data import DataLoader, Dataset

df = pd.read_csv("/workspace/CIC-IDS18/03-02-2018.csv")

from df2018_to_2017 import convert_to_second_format
df_to_2017 = convert_to_second_format(df)

from dataset_transform import zeek_snort_only, del_cheater_features
df_t = zeek_snort_only(df_to_2017)
df_t = del_cheater_features(df_t)

models_names = os.listdir("/workspace/files/neuro_invasion_detector/models3")

class MegaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = []
        for model in models_names:
            try:
                self.models.append(torch.load(os.path.join( f'/workspace/files/neuro_invasion_detector/models3/{model}', 'transformer_model.pth'), weights_only=False))
            except Exception:
                pass
    
    def forward(self, x):
        answers = []
        assurance = []
        for model in self.models:
            out = model(x)
            max_values, _ = torch.max(nn.functional.softmax(out, dim=1), dim=1)
            assurance.append(max_values)
            _, predicted = torch.max(out.data, 1)
            answers.append(predicted)

        answers = torch.stack(answers, dim=0).t()
        assurance = torch.stack(assurance, dim=0).t() 

        return answers, assurance
    
mega_model = MegaModel()



class TrafficDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = y
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Берем окно от idx до idx + seq_len
        return self.X[idx : idx + self.seq_len]


def test_mega_model(MODEL_SAVE_DIR, df: pd.DataFrame, device='cuda'):
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values

    SEQ_LEN = 20       # Длина окна (сколько пакетов назад смотрим)
    BATCH_SIZE = 1024  # Большой батч для мощной GPU

    # 1. Загружаем сохраненные объекты
    loaded_scaler = joblib.load(os.path.join(MODEL_SAVE_DIR, 'scaler.pkl'))
    #encoders = []
    #for model in models_names:
    #    print(joblib.load(os.path.join(f'/workspace/files/neuro_invasion_detector/models3/{model}', 'encoder.pkl')))
    #    encoders.append(joblib.load(os.path.join(f'/workspace/files/neuro_invasion_detector/models3/{model}', 'encoder.pkl')))
    #    #print(encoders)


    X_test = loaded_scaler.transform(X_test)
    #y_test = loaded_encoder.transform(y_test)

    NUM_CLASSES = len(np.unique(y_test))
    NUM_FEATURES = X_test.shape[1]

    print(f"Признаков: {NUM_FEATURES}, Классов: {NUM_CLASSES}")

    test_dataset = TrafficDataset(X_test, y_test, SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for inputs in tqdm(test_loader):
            inputs = inputs.to(device)
            answers, assurance = mega_model(inputs)
            for iter, (i, j) in enumerate(zip(answers, assurance)):
                if not (all(i == 0)):
                    print(iter)
                    mask = i != 0  
                    mask.to(device)
                    valid_scores = j[mask]     

                    valid_indices = torch.arange(len(j)).to(device)[mask]  
                    best_local_idx = torch.argmax(valid_scores)  # индекс в valid_scores → 0
                    best_global_idx = valid_indices[best_local_idx]
                    #loaded_encoder = encoders[best_global_idx.item()]

                    #print(loaded_encoder.inverse_transform(i[best_global_idx].item()))
                    print(f"Model number: {best_global_idx.item()}, answer: {i[best_global_idx].item()}, assurance: {j[best_global_idx].item()}, y: {y_test[iter]}")
                else:
                    continue
                    print("Benign")
            

test_mega_model(f'/workspace/files/neuro_invasion_detector/models3/{'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX'}', df_t)