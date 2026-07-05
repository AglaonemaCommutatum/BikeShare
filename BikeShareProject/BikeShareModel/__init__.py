import pandas as pd
from .data_loader import load_data
from .predictor import train_and_predict
from .scheduler import run_opt

def run():
    print("=== Bike Share Project Running ===")
    
    # Task 1
    print(">>> [Task 1] Running Prediction Model (XGBoost 800)...")
    df_train, df_test = load_data()
    if df_train is None: return
    
    y_pred = train_and_predict(df_train, df_test)
    print(f"✅ Prediction Completed. Total samples: {len(y_pred)}")
    
    # Task 2
    print(">>> [Task 2] Running Scheduling Optimization...")
    daily_demand = y_pred[:24] # 取一天
    x_opt, l_opt, rate = run_opt(daily_demand)
    
    if x_opt is not None:
        print(f"✅ Optimization Completed.")
        print(f"📊 Final User Satisfaction Rate: {rate:.2f}%")
        print(f"   (Calculated using Linear Programming)")
    else:
        print("❌ Optimization Failed.")
