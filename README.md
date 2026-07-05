# 共享单车需求预测与调度优化系统

## 项目简介

本项目针对共享单车运营场景，实现了**需求预测**与**运力调度优化**两大核心功能：

1. 基于XGBoost回归模型预测不同时段的共享单车需求量
2. 采用线性规划（Linear Programming）方法优化单车调度策略，最大化用户满意度并最小化运营成本

## 项目结构

```
BikeShareProject/
├── BikeShareModel/          # 核心功能模块
│   ├── __init__.py          # 主执行逻辑
│   ├── config.py            # 配置文件（路径/调度参数）
│   ├── data_loader.py       # 数据加载与特征工程
│   ├── predictor.py         # XGBoost需求预测模型
│   └── scheduler.py         # 线性规划调度优化
├── main.py                  # 项目入口文件
└── compare.py               # 模型对比实验（XGBoost vs 随机森林）
```

## 环境依赖

```bash
# 核心依赖包
pip install pandas numpy scipy scikit-learn xgboost
```

## 数据准备

1. 下载共享单车数据集（train.csv / test.csv）
2. 修改 `BikeShareModel/config.py` 中的路径配置：

```python
TRAIN_PATH = r"你的训练集路径/train.csv"
TEST_PATH = r"你的测试集路径/test.csv"
```

## 核心参数说明（config.py）

| 参数名     | 说明         | 默认值 |
| ---------- | ------------ | ------ |
| LIMIT_HOUR | 小时运力上限 | 250    |
| SUPPLY_0   | 初始库存     | 50     |
| C_TRANS    | 单车运输成本 | 2      |
| C_LOSS     | 缺车罚款成本 | 25     |

## 运行方式

### 1. 主程序（预测+调度）

```bash
python main.py
```

执行流程：

- 加载并预处理数据
- 训练XGBoost模型并预测24小时需求量
- 线性规划优化调度策略
- 输出用户满意度及调度结果

### 2. 模型对比实验

```bash
python compare.py
```

功能：对比XGBoost与随机森林模型的预测效果（RMSE/MAE/MAPE指标）

## 核心功能详解

### 1. 需求预测模块（predictor.py）

- 特征工程：提取时间特征（小时/月份/星期）、天气特征、高峰时段特征（I_peak）
- 模型：XGBoost回归（800棵树，学习率0.03）
- 处理：对数变换消除数据偏态，预测结果还原为原始尺度

### 2. 调度优化模块（scheduler.py）

#### 优化目标

最小化总运营成本 = 运输成本 × 调度量 + 缺车罚款 × 缺车数量

#### 约束条件

- 总调度量不超过总运力上限
- 各时段库存满足累计需求约束
- 小时调度量不超过单小时运力上限

#### 输出结果

- 各小时最优调度量（x_opt）
- 各小时缺车数量（l_opt）
- 用户满意度（1 - 总缺车数/总需求数）

## 关键指标

- **用户满意度**：优先筛选≥90%满意度的运力配置，无满足条件时取最优解
- **模型评估**：RMSE/MAE/MAPE（对比实验）

## 运行结果示例

```
=== Bike Share Project Running ===
>>> [Task 1] Running Prediction Model (XGBoost 800)...
✅ Prediction Completed. Total samples: 6493
>>> [Task 2] Running Scheduling Optimization...
✅ Optimization Completed.
📊 Final User Satisfaction Rate: 92.56%
   (Calculated using Linear Programming)
```
