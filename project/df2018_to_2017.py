import pandas as pd

# Создаем словари соответствия в обе стороны

# 1. Из второго набора в первый (как в предыдущем ответе)
second_to_first = {
    ' Destination Port': 'Dst Port',
    ' Flow Duration': 'Flow Duration',
    ' Total Fwd Packets': 'Tot Fwd Pkts',
    ' Total Backward Packets': 'Tot Bwd Pkts',
    'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
    ' Total Length of Bwd Packets': 'TotLen Bwd Pkts',
    ' Fwd Packet Length Max': 'Fwd Pkt Len Max',
    ' Fwd Packet Length Min': 'Fwd Pkt Len Min',
    ' Fwd Packet Length Mean': 'Fwd Pkt Len Mean',
    ' Fwd Packet Length Std': 'Fwd Pkt Len Std',
    'Bwd Packet Length Max': 'Bwd Pkt Len Max',
    ' Bwd Packet Length Min': 'Bwd Pkt Len Min',
    ' Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
    ' Bwd Packet Length Std': 'Bwd Pkt Len Std',
    'Flow Bytes/s': 'Flow Byts/s',
    ' Flow Packets/s': 'Flow Pkts/s',
    ' Flow IAT Mean': 'Flow IAT Mean',
    ' Flow IAT Std': 'Flow IAT Std',
    ' Flow IAT Max': 'Flow IAT Max',
    ' Flow IAT Min': 'Flow IAT Min',
    'Fwd IAT Total': 'Fwd IAT Tot',
    ' Fwd IAT Mean': 'Fwd IAT Mean',
    ' Fwd IAT Std': 'Fwd IAT Std',
    ' Fwd IAT Max': 'Fwd IAT Max',
    ' Fwd IAT Min': 'Fwd IAT Min',
    'Bwd IAT Total': 'Bwd IAT Tot',
    ' Bwd IAT Mean': 'Bwd IAT Mean',
    ' Bwd IAT Std': 'Bwd IAT Std',
    ' Bwd IAT Max': 'Bwd IAT Max',
    ' Bwd IAT Min': 'Bwd IAT Min',
    'Fwd PSH Flags': 'Fwd PSH Flags',
    ' Bwd PSH Flags': 'Bwd PSH Flags',
    ' Fwd URG Flags': 'Fwd URG Flags',
    ' Bwd URG Flags': 'Bwd URG Flags',
    ' Fwd Header Length': 'Fwd Header Len',
    ' Bwd Header Length': 'Bwd Header Len',
    'Fwd Packets/s': 'Fwd Pkts/s',
    ' Bwd Packets/s': 'Bwd Pkts/s',
    ' Min Packet Length': 'Pkt Len Min',
    ' Max Packet Length': 'Pkt Len Max',
    ' Packet Length Mean': 'Pkt Len Mean',
    ' Packet Length Std': 'Pkt Len Std',
    ' Packet Length Variance': 'Pkt Len Var',
    'FIN Flag Count': 'FIN Flag Cnt',
    ' SYN Flag Count': 'SYN Flag Cnt',
    ' RST Flag Count': 'RST Flag Cnt',
    ' PSH Flag Count': 'PSH Flag Cnt',
    ' ACK Flag Count': 'ACK Flag Cnt',
    ' URG Flag Count': 'URG Flag Cnt',
    ' CWE Flag Count': 'CWE Flag Count',
    ' ECE Flag Count': 'ECE Flag Cnt',
    ' Down/Up Ratio': 'Down/Up Ratio',
    ' Average Packet Size': 'Pkt Size Avg',
    ' Avg Fwd Segment Size': 'Fwd Seg Size Avg',
    ' Avg Bwd Segment Size': 'Bwd Seg Size Avg',
    ' Fwd Header Length.1': 'Fwd Byts/b Avg',  # Особый случай - дубликат
    'Fwd Avg Bytes/Bulk': 'Fwd Byts/b Avg',
    ' Fwd Avg Packets/Bulk': 'Fwd Pkts/b Avg',
    ' Fwd Avg Bulk Rate': 'Fwd Blk Rate Avg',
    ' Bwd Avg Bytes/Bulk': 'Bwd Byts/b Avg',
    ' Bwd Avg Packets/Bulk': 'Bwd Pkts/b Avg',
    'Bwd Avg Bulk Rate': 'Bwd Blk Rate Avg',
    'Subflow Fwd Packets': 'Subflow Fwd Pkts',
    ' Subflow Fwd Bytes': 'Subflow Fwd Byts',
    ' Subflow Bwd Packets': 'Subflow Bwd Pkts',
    ' Subflow Bwd Bytes': 'Subflow Bwd Byts',
    'Init_Win_bytes_forward': 'Init Fwd Win Byts',
    ' Init_Win_bytes_backward': 'Init Bwd Win Byts',
    ' act_data_pkt_fwd': 'Fwd Act Data Pkts',
    ' min_seg_size_forward': 'Fwd Seg Size Min',
    'Active Mean': 'Active Mean',
    ' Active Std': 'Active Std',
    ' Active Max': 'Active Max',
    ' Active Min': 'Active Min',
    'Idle Mean': 'Idle Mean',
    ' Idle Std': 'Idle Std',
    ' Idle Max': 'Idle Max',
    ' Idle Min': 'Idle Min',
    ' Label': 'Label'
}

# 2. Из первого набора во второй (обратное преобразование)
first_to_second = {
    'Dst Port': ' Destination Port',
    'Flow Duration': ' Flow Duration',
    'Tot Fwd Pkts': ' Total Fwd Packets',
    'Tot Bwd Pkts': ' Total Backward Packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts': ' Total Length of Bwd Packets',
    'Fwd Pkt Len Max': ' Fwd Packet Length Max',
    'Fwd Pkt Len Min': ' Fwd Packet Length Min',
    'Fwd Pkt Len Mean': ' Fwd Packet Length Mean',
    'Fwd Pkt Len Std': ' Fwd Packet Length Std',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max',
    'Bwd Pkt Len Min': ' Bwd Packet Length Min',
    'Bwd Pkt Len Mean': ' Bwd Packet Length Mean',
    'Bwd Pkt Len Std': ' Bwd Packet Length Std',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Pkts/s': ' Flow Packets/s',
    'Flow IAT Mean': ' Flow IAT Mean',
    'Flow IAT Std': ' Flow IAT Std',
    'Flow IAT Max': ' Flow IAT Max',
    'Flow IAT Min': ' Flow IAT Min',
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Fwd IAT Mean': ' Fwd IAT Mean',
    'Fwd IAT Std': ' Fwd IAT Std',
    'Fwd IAT Max': ' Fwd IAT Max',
    'Fwd IAT Min': ' Fwd IAT Min',
    'Bwd IAT Tot': 'Bwd IAT Total',
    'Bwd IAT Mean': ' Bwd IAT Mean',
    'Bwd IAT Std': ' Bwd IAT Std',
    'Bwd IAT Max': ' Bwd IAT Max',
    'Bwd IAT Min': ' Bwd IAT Min',
    'Fwd PSH Flags': 'Fwd PSH Flags',
    'Bwd PSH Flags': ' Bwd PSH Flags',
    'Fwd URG Flags': ' Fwd URG Flags',
    'Bwd URG Flags': ' Bwd URG Flags',
    'Fwd Header Len': ' Fwd Header Length',
    'Bwd Header Len': ' Bwd Header Length',
    'Fwd Pkts/s': 'Fwd Packets/s',
    'Bwd Pkts/s': ' Bwd Packets/s',
    'Pkt Len Min': ' Min Packet Length',
    'Pkt Len Max': ' Max Packet Length',
    'Pkt Len Mean': ' Packet Length Mean',
    'Pkt Len Std': ' Packet Length Std',
    'Pkt Len Var': ' Packet Length Variance',
    'FIN Flag Cnt': 'FIN Flag Count',
    'SYN Flag Cnt': ' SYN Flag Count',
    'RST Flag Cnt': ' RST Flag Count',
    'PSH Flag Cnt': ' PSH Flag Count',
    'ACK Flag Cnt': ' ACK Flag Count',
    'URG Flag Cnt': ' URG Flag Count',
    'CWE Flag Count': ' CWE Flag Count',
    'ECE Flag Cnt': ' ECE Flag Count',
    'Down/Up Ratio': ' Down/Up Ratio',
    'Pkt Size Avg': ' Average Packet Size',
    'Fwd Seg Size Avg': ' Avg Fwd Segment Size',
    'Bwd Seg Size Avg': ' Avg Bwd Segment Size',
    'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',  # Специальное отображение
    'Fwd Pkts/b Avg': ' Fwd Avg Packets/Bulk',
    'Fwd Blk Rate Avg': ' Fwd Avg Bulk Rate',
    'Bwd Byts/b Avg': ' Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg': ' Bwd Avg Packets/Bulk',
    'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
    'Subflow Fwd Pkts': 'Subflow Fwd Packets',
    'Subflow Fwd Byts': ' Subflow Fwd Bytes',
    'Subflow Bwd Pkts': ' Subflow Bwd Packets',
    'Subflow Bwd Byts': ' Subflow Bwd Bytes',
    'Init Fwd Win Byts': 'Init_Win_bytes_forward',
    'Init Bwd Win Byts': ' Init_Win_bytes_backward',
    'Fwd Act Data Pkts': ' act_data_pkt_fwd',
    'Fwd Seg Size Min': ' min_seg_size_forward',
    'Active Mean': 'Active Mean',
    'Active Std': ' Active Std',
    'Active Max': ' Active Max',
    'Active Min': ' Active Min',
    'Idle Mean': 'Idle Mean',
    'Idle Std': ' Idle Std',
    'Idle Max': ' Idle Max',
    'Idle Min': ' Idle Min',
    'Label': ' Label'
}

# Для особых случаев (дубликатов во втором наборе)
# Когда 'Fwd Byts/b Avg' может соответствовать двум колонкам во втором наборе
special_cases_forward = {
    'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk'  # Выбираем более подходящее
}

# Функции преобразования

def convert_to_first_format(df):
    """
    Преобразует DataFrame с колонками из второго формата в первый формат.
    
    Parameters:
    df (pd.DataFrame): DataFrame с колонками второго формата
    
    Returns:
    pd.DataFrame: DataFrame с колонками первого формата
    """
    # Создаем копию для безопасности
    df_result = df.copy()
    
    # Функция для очистки названий колонок от лишних пробелов
    def clean_column_name(col):
        return col.strip() if isinstance(col, str) else col
    
    # Создаем очищенный словарь для поиска
    clean_mapping = {}
    for old_name, new_name in second_to_first.items():
        clean_old = clean_column_name(old_name)
        clean_mapping[clean_old] = new_name
    
    # Переименовываем колонки
    rename_dict = {}
    for col in df_result.columns:
        clean_col = clean_column_name(col)
        if clean_col in clean_mapping:
            rename_dict[col] = clean_mapping[clean_col]
    
    if rename_dict:
        df_result = df_result.rename(columns=rename_dict)
    
    return df_result

def convert_to_second_format(df, handle_duplicates='first'):
    """
    Преобразует DataFrame с колонками из первого формата во второй формат.
    
    Parameters:
    df (pd.DataFrame): DataFrame с колонками первого формата
    handle_duplicates (str): Как обрабатывать дубликаты при обратном преобразовании
        'first' - использовать первое соответствие (по умолчанию)
        'both' - создать обе колонки (может привести к дубликатам)
    
    Returns:
    pd.DataFrame: DataFrame с колонками второго формата
    """
    # Создаем копию для безопасности
    df_result = df.copy()
    
    if handle_duplicates == 'both':
        # Для особых случаев создаем дополнительную колонку
        # 'Fwd Byts/b Avg' соответствует 'Fwd Avg Bytes/Bulk' и ' Fwd Header Length.1'
        if 'Fwd Byts/b Avg' in df_result.columns:
            # Переименовываем основную колонку
            df_result = df_result.rename(columns={'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk'})
            # Создаем копию для дубликата
            df_result[' Fwd Header Length.1'] = df_result['Fwd Avg Bytes/Bulk'].copy()
    
    # Основное переименование
    rename_dict = {}
    for col in df_result.columns:
        if col in first_to_second:
            rename_dict[col] = first_to_second[col]
        elif col in special_cases_forward:
            rename_dict[col] = special_cases_forward[col]
    
    if rename_dict:
        df_result = df_result.rename(columns=rename_dict)
    
    return df_result

def get_column_conversion_info():
    """
    Возвращает информацию о преобразовании колонок.
    """
    first_set = [
        'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts',
        'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
        'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
        'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
        'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
        'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
        'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
        'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
        'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
        'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
        'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
        'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
        'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
        'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
        'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
        'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
        'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
        'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
        'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
        'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label'
    ]
    
    second_set = [
        ' Destination Port', ' Flow Duration', ' Total Fwd Packets',
        ' Total Backward Packets', 'Total Length of Fwd Packets',
        ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
        ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
        ' Fwd Packet Length Std', 'Bwd Packet Length Max',
        ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
        ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s',
        ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
        'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
        ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
        ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
        ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
        ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
        ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
        ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
        ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
        ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
        ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
        ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
        ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
        ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
        'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
        ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
        ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
        ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
        ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min',
        ' Label'
    ]
    
    return {
        'first_set': first_set,
        'second_set': second_set,
        'first_to_second': first_to_second,
        'second_to_first': second_to_first,
        'missing_in_second': ['Protocol', 'Timestamp'],
        'duplicates_in_second': ['Fwd Header Length.1', 'Fwd Avg Bytes/Bulk']
    }

# Пример использования
if __name__ == "__main__":
    # Создаем тестовые DataFrames
    # Пример с колонками из второго набора
    second_format_columns = [' Destination Port', ' Flow Duration', ' Total Fwd Packets']
    df_second = pd.DataFrame(columns=second_format_columns)
    
    # Преобразуем в первый формат
    df_first = convert_to_first_format(df_second)
    print("Первый формат колонок:", list(df_first.columns))
    
    # Преобразуем обратно во второй формат
    df_second_again = convert_to_second_format(df_first)
    print("Второй формат колонок:", list(df_second_again.columns))
    
    # Получить информацию о преобразовании
    info = get_column_conversion_info()
    print(f"Колонки, отсутствующие во втором наборе: {info['missing_in_second']}")
    print(f"Дубликаты во втором наборе: {info['duplicates_in_second']}")