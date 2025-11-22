import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import torch
import os

def del_cheater_features(df):
    # Список опасных признаков (утечка данных)
    dangerous_features = [
        # Потоковые скорости (нельзя знать онлайн)
        'Flow Bytes/s', ' Flow Packets/s', 
        'Fwd Packets/s', ' Bwd Packets/s',

        # Bulk features (искусственные признаки CICFlowMeter)
        'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
        ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
        ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',

        # Subflows (искусственные признаки CICFlowMeter)
        'Subflow Fwd Packets', ' Subflow Fwd Bytes',
        ' Subflow Bwd Packets', ' Subflow Bwd Bytes',

        # Active / Idle (требует знания полного потока)
        'Active Mean', ' Active Std', ' Active Max', ' Active Min',
        'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min',

        # Дубликаты и ошибки CICFlowMeter
        ' Fwd Header Length.1'
    ]

    # Опционально: удалить TCP Window Size (часто вызывает косвенную утечку)
    # dangerous_features += [
    #     'Init_Win_bytes_forward', ' Init_Win_bytes_backward'
    # ]

    # Удаляем только те признаки, которые есть в датафрейме
    existing_dangerous = [c for c in dangerous_features if c in df.columns]

    df = df.drop(columns=existing_dangerous)
    
    print("Удалено опасных признаков:", len(existing_dangerous))
    print("Финальное число признаков:", df.shape[1])

    return df

def zeek_snort_only(df):
    zeek_snort_columns = [
        ' Destination Port', ' Flow Duration',

        ' Total Fwd Packets', ' Total Backward Packets',
        'Total Length of Fwd Packets', ' Total Length of Bwd Packets',

        ' Fwd Packet Length Max', ' Fwd Packet Length Min',
        ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
        'Bwd Packet Length Max', ' Bwd Packet Length Min',
        ' Bwd Packet Length Mean', ' Bwd Packet Length Std',

        ' Min Packet Length', ' Max Packet Length',
        ' Packet Length Mean', ' Packet Length Std',
        ' Packet Length Variance', ' Average Packet Size',

        ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
        'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min',
        'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',

        'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count',
        ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
        ' CWE Flag Count', ' ECE Flag Count',

        ' Fwd Header Length', ' Bwd Header Length',
        'min_seg_size_forward',

        'Init_Win_bytes_forward', ' Init_Win_bytes_backward',

        'Active Mean', ' Active Std', ' Active Max', ' Active Min',
        'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min',

        ' Label'  # оставить целевой признак
    ]

    # Пересечение со столбцами, которые реально есть в DataFrame
    existing_cols = [c for c in zeek_snort_columns if c in df.columns]

    # Фильтрация
    df = df[existing_cols]
    print("Готово! Число оставленных колонок:", len(df.columns))

    return df

# Класс Dataset для PyTorch (создает скользящее окно)
class TrafficDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Берем окно от idx до idx + seq_len
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len]

def get_dataset(df: pd.DataFrame, MODEL_SAVE_DIR='/'):
    df = del_cheater_features(df)
    df = zeek_snort_only(df)

    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    SEQ_LEN = 20       # Длина окна (сколько пакетов назад смотрим)
    BATCH_SIZE = 1024  # Большой батч для мощной GPU

    # Загрузка
    #print("Загрузка данных...")
    #df = pd.read_csv('/workspace/files/archive/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv') 
    #df.columns = df.columns.str.strip()
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df.dropna(inplace=True)

    # Препроцессинг
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'scaler.pkl'))

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    NUM_CLASSES = len(np.unique(y_encoded))
    NUM_FEATURES = X_scaled.shape[1]
    joblib.dump(encoder, os.path.join(MODEL_SAVE_DIR, 'encoder.pkl'))


    print(f"Признаков: {NUM_FEATURES}, Классов: {NUM_CLASSES}")

    # Создаем датасеты
    dataset = TrafficDataset(X_scaled, y_encoded, SEQ_LEN)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    return loader, (NUM_FEATURES, NUM_CLASSES)