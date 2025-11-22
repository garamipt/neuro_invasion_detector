from dataset_transform import get_dataset
from model_generator import get_model
from learning import get_trainde_model
from test_model import test_model
import os 
import pandas as pd
import torch

READ_DIR = '/workspace/files/archive'
SAVE_DIR = '/workspace/models'

skip = ['Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Friday-WorkingHours-Morning.pcap_ISCX.csv', 'Monday-WorkingHours.pcap_ISCX.csv']


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Папка создана: {folder_path}")
    else:
        print(f"Папка уже существует: {folder_path}")


for file_name in os.listdir(READ_DIR):
    print(file_name, file_name in skip)
    if file_name in skip:
        print(file_name, 'skiped')
        continue
    create_folder_if_not_exists(f"{SAVE_DIR}/{file_name[:-4]}")

    df = pd.read_csv(f"{READ_DIR}/{file_name}")

    train_df = df[:int(len(df) * 0.9)]
    test_df = df[int(len(df) * 0.9):]
    
    train_loader, INFO = get_dataset(train_df, f"{SAVE_DIR}/{file_name[:-4]}")
    model = get_model(*INFO)
    model = get_trainde_model(model, train_loader, 15)

    torch.save(model, f"{SAVE_DIR}/{file_name[:-4]}/transformer_model.pth")
    print("Transformer модель сохранена.")

    test_model(f"{SAVE_DIR}/{file_name[:-4]}", test_df)