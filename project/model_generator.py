import torch
import torch.nn as nn
import math


class NetworkTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(NetworkTransformer, self).__init__()
        
        # Проекция входных признаков в размерность модели
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Позиционное кодирование (чтобы знать порядок пакетов)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Сам блок Трансформера (Encoder)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Выходная голова
        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.input_embedding(src) # [Batch, Seq_Len, Features] -> [Batch, Seq_Len, d_model]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        # Берем только последний временной шаг для предсказания
        output = output[:, -1, :] 
        output = self.decoder(output)
        return output

# Вспомогательный класс для позиционирования (стандартный из туториалов PyTorch)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Инициализация модели
def get_model(NUM_FEATURES, NUM_CLASSES, device='cuda'):
    model = NetworkTransformer(
        input_dim=NUM_FEATURES, 
        num_classes=NUM_CLASSES,
        d_model=128,  
        nhead=8,      
        num_layers=3  
    ).to(device)

    return model