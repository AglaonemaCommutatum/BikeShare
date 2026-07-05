import xgboost as xgb
import numpy as np
import pandas as pd
from . import data_loader

def train_and_predict(df_train, df_test):
    df_train_feat = data_loader.extract_features(df_train)
    df_test_feat = data_loader.extract_features(df_test)
    
    features = ['season', 'holiday', 'workingday', 'weather', 
                'temp', 'atemp', 'humidity', 'windspeed', 
                'hour', 'month', 'weekday', 'temp_diff', 'I_peak']
                
    X_train = df_train_feat[features]
    y_train = np.log1p(df_train_feat['count'])
    X_test = df_test_feat[features]
    
    # ⚠️ 实际应用参数：800次迭代，0.03学习率
    model = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=7,
        objective='reg:squarederror',
        n_jobs=-1,
        enable_categorical=True
    )
    model.fit(X_train, y_train)
    
    log_pred = model.predict(X_test)
    y_pred = np.maximum(np.expm1(log_pred), 0)
    
    return y_pred
