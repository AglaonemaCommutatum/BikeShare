import numpy as np
import pandas as pd
from scipy.optimize import linprog
from . import config

def run_opt(demand_t):
    T = 24
    results = []
    
    # 1. 灵敏度分析
    for total_supply in range(1500, 5500, 100):
        c = np.array([config.C_TRANS]*T + [config.C_LOSS]*T)
        A_ub = []; b_ub = []
        
        row_total = np.zeros(2*T); row_total[:T] = 1
        A_ub.append(row_total); b_ub.append(total_supply)
        
        cum_demand = 0
        for t in range(T):
            cum_demand += demand_t[t]
            row = np.zeros(2*T); row[:t+1] = -1; row[T:T+t+1] = -1
            A_ub.append(row); b_ub.append(config.SUPPLY_0 - cum_demand)

        bounds = [(0, config.LIMIT_HOUR) for _ in range(T)] + [(0, demand_t[t]) for t in range(T)]
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
        
        if res.success:
            sat = (1 - (sum(res.x[T:]) / sum(demand_t))) * 100
            results.append({'Supply': total_supply, 'Satisfaction': sat})

    # 2. 选推荐点 (>90%)
    df = pd.DataFrame(results)
    qualified = df[df['Satisfaction'] >= 90]
    if not qualified.empty:
        rec_supply = int(qualified.iloc[0]['Supply'])
        rec_rate = qualified.iloc[0]['Satisfaction']
    else:
        best = df.loc[df['Satisfaction'].idxmax()]
        rec_supply = int(best['Supply'])
        rec_rate = best['Satisfaction']

    # 3. 最终计算详情
    c = np.array([config.C_TRANS]*T + [config.C_LOSS]*T)
    A_ub = []; b_ub = []
    row_total = np.zeros(2*T); row_total[:T] = 1
    A_ub.append(row_total); b_ub.append(rec_supply)
    cum_demand = 0
    for t in range(T):
        cum_demand += demand_t[t]
        row = np.zeros(2*T); row[:t+1] = -1; row[T:T+t+1] = -1
        A_ub.append(row); b_ub.append(config.SUPPLY_0 - cum_demand)
    bounds = [(0, config.LIMIT_HOUR) for _ in range(T)] + [(0, demand_t[t]) for t in range(T)]
    
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
    
    if res.success:
        x_opt = np.round(res.x[:T])
        l_opt = np.round(res.x[T:])
        return x_opt, l_opt, rec_rate
    return None, None, 0
