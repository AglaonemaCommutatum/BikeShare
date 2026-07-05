import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

TRAIN_PATH = r"D:\cxdownload\bike-sharing-demand\train.csv"

def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

print(">>> Loading Data for Model Comparison...")
df = pd.read_csv(TRAIN_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['weekday'] = df['datetime'].dt.dayofweek
df['temp_diff'] = df['atemp'] - df['temp']

def get_peak(row):
    h=row['hour']; w=row['workingday']
    if w==1 and h in [7,8,9,17,18,19]: return 1
    elif w==0 and 10<=h<=16: return 2
    else: return 0
df['I_peak'] = df.apply(get_peak, axis=1)

X = df[['season','holiday','workingday','weather','temp','atemp',
        'humidity','windspeed','hour','month','weekday','temp_diff','I_peak']]
X = pd.get_dummies(X, columns=['I_peak'], drop_first=True)
y = np.log1p(df['count'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_true = np.expm1(y_val)

print("\n=== Model Comparison (n_estimators=1200) ===")

# 1. Random Forest
print("1. Training Random Forest (1200 trees)...")
rf = RandomForestRegressor(n_estimators=1200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = np.maximum(np.expm1(rf.predict(X_val)), 0)
print(f"   RF  -> RMSE: {np.sqrt(mean_squared_error(y_true, rf_pred)):.4f} | "
      f"MAE: {mean_absolute_error(y_true, rf_pred):.4f} | "
      f"MAPE: {mape(y_true, rf_pred):.2f}%")

# 2. XGBoost
print("2. Training XGBoost (1200 trees, lr=0.02)...")
xg = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.02, max_depth=7, 
                      subsample=0.8, random_state=42, n_jobs=-1)
xg.fit(X_train, y_train)
xg_pred = np.maximum(np.expm1(xg.predict(X_val)), 0)
print(f"   XGB -> RMSE: {np.sqrt(mean_squared_error(y_true, xg_pred)):.4f} | "
      f"MAE: {mean_absolute_error(y_true, xg_pred):.4f} | "
      f"MAPE: {mape(y_true, xg_pred):.2f}%")

print("\nComparison Done.")
input("Press Enter...")
