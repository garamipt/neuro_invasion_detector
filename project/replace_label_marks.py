import pandas as pd
import numpy as np

def convert_label_column(df, label_column='Label'):
    """
    Преобразует колонку с метками: 'benign' (в любом регистре) -> 0, все остальное -> 1.
    
    Parameters:
    df (pd.DataFrame): DataFrame с колонкой меток
    label_column (str): Название колонки с метками (по умолчанию 'Label')
    
    Returns:
    pd.DataFrame: DataFrame с преобразованной колонкой меток
    """
    # Создаем копию DataFrame для безопасности
    df_result = df.copy()
    
    # Проверяем, существует ли колонка с метками
    if label_column not in df_result.columns:
        print(f"Предупреждение: колонка '{label_column}' не найдена в DataFrame")
        print(f"Доступные колонки: {list(df_result.columns)}")
        return df_result
    
    # Проверяем тип данных в колонке
    print(f"Тип данных в колонке '{label_column}': {df_result[label_column].dtype}")
    print(f"Уникальные значения до преобразования: {df_result[label_column].unique()[:10]}")
    
    # Преобразуем строковые значения в нижний регистр и сравниваем с 'benign'
    # Для числовых значений оставляем как есть (или можно обработать отдельно)
    if df_result[label_column].dtype == 'object':
        # Для строковых значений
        # Создаем временную колонку в нижнем регистре для сравнения
        labels_lower = df_result[label_column].astype(str).str.lower().str.strip()
        
        # Преобразуем: 'benign' -> 0, остальное -> 1
        df_result[label_column] = np.where(labels_lower == 'benign', 0, 1)
        
        # Преобразуем в целочисленный тип
        df_result[label_column] = df_result[label_column].astype(int)
    else:
        # Для числовых или других типов данных
        print(f"Колонка '{label_column}' имеет тип {df_result[label_column].dtype}")
        print("Предполагается, что значения уже числовые или требуют другой обработки")
    
    print(f"Уникальные значения после преобразования: {df_result[label_column].unique()}")
    print(f"Распределение классов:")
    print(df_result[label_column].value_counts())
    
    return df_result

# Альтернативная версия с сохранением исходных значений в отдельной колонке
def convert_label_column_with_backup(df, label_column='Label', backup_suffix='_original'):
    """
    Преобразует колонку с метками, сохраняя оригинальные значения в отдельной колонке.
    """
    df_result = df.copy()
    
    if label_column not in df_result.columns:
        print(f"Предупреждение: колонка '{label_column}' не найдена")
        return df_result
    
    # Сохраняем оригинальные значения
    backup_column = f"{label_column}{backup_suffix}"
    df_result[backup_column] = df_result[label_column].copy()
    
    # Применяем преобразование
    return convert_label_column(df_result, label_column)

# Функция для проверки и обработки разных вариантов названий колонки
def find_and_convert_label(df, possible_names=['Label', 'label', 'labels', 'LABEL', 'target']):
    """
    Ищет колонку с метками по разным возможным названиям и преобразует её.
    
    Parameters:
    df (pd.DataFrame): Входной DataFrame
    possible_names (list): Список возможных названий колонки с метками
    
    Returns:
    pd.DataFrame: DataFrame с преобразованной колонкой меток
    """
    df_result = df.copy()
    
    # Ищем существующую колонку с метками
    label_col = None
    for name in possible_names:
        if name in df_result.columns:
            label_col = name
            break
    
    if label_col:
        print(f"Найдена колонка с метками: '{label_col}'")
        df_result = convert_label_column(df_result, label_col)
    else:
        print(f"Не найдена колонка с метками. Искали: {possible_names}")
        print(f"Доступные колонки: {list(df_result.columns)}")
    
    return df_result

# Пример использования
if __name__ == "__main__":
    # Создаем тестовый DataFrame
    test_data = {
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [10, 20, 30, 40, 50],
        'Label': ['Benign', 'MALWARE', 'benign', 'DoS', 'BENIGN']
    }
    
    df_test = pd.DataFrame(test_data)
    print("Исходный DataFrame:")
    print(df_test)
    print("\n" + "="*50 + "\n")
    
    # Вариант 1: Простое преобразование
    df_converted = convert_label_column(df_test)
    print("\nDataFrame после преобразования:")
    print(df_converted)
    
    # Вариант 2: С сохранением оригинальных значений
    df_with_backup = convert_label_column_with_backup(df_test)
    print("\nDataFrame с сохранением оригинальных значений:")
    print(df_with_backup)
    
    # Вариант 3: С автоматическим поиском колонки
    # Создаем DataFrame с другим названием колонки
    test_data2 = {
        'Feature1': [1, 2, 3],
        'label': ['attack', 'BENIGN', 'normal']
    }
    df_test2 = pd.DataFrame(test_data2)
    
    print("\n" + "="*50)
    print("Тест с другим названием колонки:")
    df_converted2 = find_and_convert_label(df_test2)
    print(df_converted2)

# Дополнительная функция для интеграции с предыдущим кодом переименования колонок
def process_dataset(df, convert_labels=True, label_column='Label'):
    """
    Полная обработка датасета: переименование колонок и преобразование меток.
    
    Parameters:
    df (pd.DataFrame): Исходный DataFrame
    convert_labels (bool): Нужно ли преобразовывать метки
    label_column (str): Название колонки с метками
    
    Returns:
    pd.DataFrame: Обработанный DataFrame
    """
    from your_previous_module import convert_to_first_format  # Импортируйте ваш предыдущий модуль
    
    # 1. Сначала переименовываем колонки (если нужно)
    # df = convert_to_first_format(df)  # Раскомментировать, если нужно
    
    # 2. Затем преобразуем метки
    if convert_labels:
        df = find_and_convert_label(df)
    
    return df