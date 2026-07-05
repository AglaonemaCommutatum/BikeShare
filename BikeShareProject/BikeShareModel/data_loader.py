import pandas as pd
from . import config

def load_data():
    try:
        df_train = pd.read_csv(config.TRAIN_PATH)
        df_test = pd.read_csv(config.TEST_PATH)
        df_train['datetime'] = pd.to_datetime(df_train['datetime'])
        df_test['datetime'] = pd.to_datetime(df_test['datetime'])
        return df_train, df_test
    except:
        print("❌ 数据读取失败，请检查路径")
        return None, None

def extract_features(df):
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.dayofweek
    df['workingday'] = df['workingday']
    
    if 'atemp' in df.columns:
        df['temp_diff'] = df['atemp'] - df['temp']
    else:
        df['temp_diff'] = 0
        
    # 核心特征 I_peak
    def get_peak_type(row):
        h = row['hour']; w = row['workingday']
        if w == 1 and (h in [7, 8, 9, 17, 18, 19]): return 1 
        elif w == 0 and (10 <= h <= 16): return 2
        else: return 0

    df['I_peak'] = df.apply(get_peak_type, axis=1)
    df['I_peak'] = df['I_peak'].astype('category')
    return df
