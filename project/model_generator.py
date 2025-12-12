"""
model_generator.py

Этот файл содержит реализацию трансформерной нейросети для задач классификации сетевого трафика.
Добавлены подробные докстринги и комментарии на русском языке.
"""

import torch
import torch.nn as nn
import math


class NetworkTransformer(nn.Module):
    """
    Класс нейронной сети на основе Transformer Encoder для анализа последовательностей
    сетевых пакетов. Используется для задач классификации (например, выявления атак).

    Параметры:
        input_dim (int): количество входных признаков на один пакет.
        num_classes (int): количество классов для классификации.
        d_model (int): размерность скрытого представления внутри трансформера.
        nhead (int): количество голов в multi-head attention.
        num_layers (int): число слоёв Transformer Encoder.
        dropout (float): вероятность dropout в блоках модели.

    Архитектура:
        - Линейная проекция входных признаков в пространство размерности d_model.
        - Позиционное кодирование (PositionalEncoding), добавляющее информацию о порядке.
        - Несколько слоёв TransformerEncoder.
        - Линейный классификатор, принимающий последнюю временную точку выхода.
    """

    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(NetworkTransformer, self).__init__()

        # Проекция входных признаков в размерность модели
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Позиционное кодирование (чтобы модель учитывала порядок пакетов)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Слои Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,  # скрытые размеры внутри FFN блока
            dropout=dropout,
            batch_first=True      # важно: вход имеет форму (batch, seq, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Выходная голова классификатора
        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, src):
        """
        Прямой проход (inference) модели.

        Параметры:
            src (Tensor): входной тензор размера [Batch, Seq_Len, input_dim]

        Возвращает:
            Tensor: выход размерности [Batch, num_classes]
        """
        # Встраиваем вход в пространство модельного размера
        src = self.input_embedding(src)

        # Добавляем позиционное кодирование
        src = self.pos_encoder(src)

        # Прогон через Transformer Encoder
        output = self.transformer_encoder(src)

        # Берём только последний временной шаг (обычный подход для классификации последовательностей)
        output = output[:, -1, :]

        # Применяем линейный классификатор
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    """
    Позиционное кодирование, добавляющее в последовательность информацию
    о положении каждого элемента. Классический вариант, используемый в Transformer.

    Параметры:
        d_model (int): размерность векторного представления.
        dropout (float): вероятность dropout после добавления кодировки.
        max_len (int): максимальная длина последовательности, для которой
                       генерируются синусоидальные позиционные векторы.

    Механизм:
        Генерирует двумерную матрицу PE[max_len, d_model], где синусы используются
        для чётных индексов, а косинусы — для нечётных.

        Эти значения добавляются ко входному тензору X.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Матрица позиционных кодировок
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Синус для чётных индексов
        pe[:, 0::2] = torch.sin(position * div_term)
        # Косинус для нечётных
        pe[:, 1::2] = torch.cos(position * div_term)

        # Регистрируем как не обучаемый буфер — он будет сохраняться вместе с моделью
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        """
        Применяет позиционное кодирование к входному тензору.

        Параметры:
            x (Tensor): входной тензор [Batch, Seq_Len, d_model]

        Возвращает:
            Tensor: тот же тензор + позиционное смещение.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def get_model(NUM_FEATURES, NUM_CLASSES, device ='cuda'):
    """
    Фабричная функция для создания экземпляра NetworkTransformer.

    Параметры:
        NUM_FEATURES (int): количество входных признаков.
        NUM_CLASSES (int): количество классов.
        device (str): устройство ('cpu' или 'cuda').

    Возвращает:
        NetworkTransformer: инициализированная модель, перенесённая на заданное устройство.
    """
    model = NetworkTransformer(
        input_dim=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        d_model=128,
        nhead=8,
        num_layers=3
    ).to(device)

    return model