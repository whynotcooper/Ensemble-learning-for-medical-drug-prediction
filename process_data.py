import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
import gc
import random
import math
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import log_loss

import category_encoders as ce
from sklearn.feature_selection import VarianceThreshold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")
def get_logger(filename='log'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
logger = get_logger()
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
def cate2num(df):
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72: 2})
    df['cp_dose'] = df['cp_dose'].map({'D1': 3, 'D2': 4})
    return df

def preprocess_data(test_df):
        # 过滤掉 'ctl_vehicle' 类型的行
        test = test_df[test_df['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

        # 定义分类特征和数值特征
        cat_features = ['cp_time', 'cp_dose']
        num_features = [c for c in test.columns if test.dtypes[c] != 'object']
        num_features = [c for c in num_features if c not in cat_features]

        # 将分类特征转换为数值
        test = cate2num(test)

        # 提取数值特征
        numeric_data = test[num_features]

        return test


class TestDataset(Dataset):
    def __init__(self, df, num_features, cat_features):
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values

    def __len__(self):
        return len(self.cont_values)

    def __getitem__(self, idx):
        cont_x = torch.FloatTensor(self.cont_values[idx])
        cate_x = torch.LongTensor(self.cate_values[idx])

        return cont_x, cate_x


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size,dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
                          nn.Linear(len(num_features), hidden_size),
                          nn.BatchNorm1d(hidden_size),
                          nn.Dropout(dropout),
                          nn.PReLU(),
                          nn.Linear(hidden_size, hidden_size),
                          nn.BatchNorm1d(hidden_size),
                          nn.Dropout(dropout),
                          nn.PReLU(),
                          nn.Linear(hidden_size, num_targets),
                          )

    def forward(self, cont_x, cate_x):
        # no use of cate_x yet
        x = self.mlp(cont_x)
        return x



import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def run_single_nn(cfg, test, num_features, cat_features, device, fold_num=0, seed=42):
    # 创建测试数据集
    test_dataset = TestDataset(test, num_features, cat_features)

    # 打印配置信息
    print(cfg)

    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=4, pin_memory=True)

    # 定义模型
    model = Model(num_features=cfg["num_features"], num_targets=cfg["num_targets"], hidden_size=cfg["hidden_size"], dropout=cfg["dropout"])

    # 加载权重
    weights_path = f"input/fold0_seed1.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 将模型移动到指定设备
    model.to(device)

    # 设置模型为评估模式
    model.eval()

    # 进行推理
    predictions = inference_fn(test_loader, model, device)

    # 清理 GPU 缓存
    torch.cuda.empty_cache()

    return predictions


# 示例推理函数
def inference_fn(test_loader, model, device):
    model.eval()
    preds = []

    for step, (cont_x, cate_x) in enumerate(test_loader):
        cont_x, cate_x = cont_x.to(device), cate_x.to(device)

        with torch.no_grad():
            pred = model(cont_x, cate_x)
            pred = pred.view(pred.shape[0], -1)
        preds.append(pred.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)
    return preds
import os
from datetime import datetime

def process_data(cfg, test,  num_features, cat_features, device, fold_num=0, seed=42):
    # 运行单次神经网络训练并获取预测结果
    predictions = run_single_nn(cfg, test, num_features, cat_features, device, fold_num=fold_num, seed=seed)
    # 将预测结果保存到 test 数据框中
    if isinstance(predictions, np.ndarray):
        predictions = pd.DataFrame(predictions, columns=[f"target_{i}" for i in range(predictions.shape[1])])

    # 创建保存目录（如果不存在）
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    # 生成带有时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{"pred"}_{timestamp}.csv"

    # 定义保存文件的路径
    pred_file_path = os.path.join(output_dir, filename)

    if isinstance(predictions, np.ndarray):
        predictions = pd.DataFrame(predictions, columns=[f"target_{i}" for i in range(predictions.shape[1])])

        # 保存预测结果到 CSV 文件
    predictions.to_csv(pred_file_path, index=False)

    # 返回保存的文件路径
    return pred_file_path
    # 保存预测结果到 CSV 文件
import matplotlib.pyplot as plt
def process_data2(cfg, test, num_features, cat_features, device, fold_num=0, seed=42):
    # 运行单次神经网络训练并获取预测结果
    predictions = run_single_nn(cfg, test, num_features, cat_features, device, fold_num=fold_num, seed=seed)
    # 将预测结果保存到 test 数据框中
    if isinstance(predictions, np.ndarray):
        predictions = pd.DataFrame(predictions, columns=[f"target_{i}" for i in range(predictions.shape[1])])

    # 创建保存目录（如果不存在）
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    # 生成带有时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 假设 pred 是 h×w 的 nd_array 数组
    pred = predictions.values  # 获取预测结果的数值部分

    # 对每一行求前五个最大的数目及其索引
    top5_values = []
    top5_indices = []
    for row in pred:
        # 获取一行中前五个最大的值及其索引
        top5_idx = np.argsort(row)[-5:][::-1]  # 获取前五个最大值的索引，并按降序排列
        top5_val = row[top5_idx]
        top5_values.append(top5_val)
        top5_indices.append(top5_idx)

    # 将结果转换为 DataFrame
    top5_df = pd.DataFrame(top5_values, columns=[f"top_{i+1}_value" for i in range(5)])
    top5_indices_df = pd.DataFrame(top5_indices, columns=[f"top_{i+1}_index" for i in range(5)])

    # 绘制每一行五个数的柱状图并保存
    for i, (vals, idxs) in enumerate(zip(top5_values, top5_indices)):
        plt.figure(figsize=(10, 6))
        plt.bar(range(5), vals)
        plt.xlabel('Top 5 Indices')
        plt.ylabel('Values')
        plt.title(f'Top 5 Values for Row {i+1}')
        plt.xticks(range(5), idxs)
        # 保存柱状图
        chart_filename = f"{output_dir}/row_{i+1}_top5_{timestamp}.png"
        plt.savefig(chart_filename)
        plt.close()

    # 保存返回结果
    # 保存 top5_df 和 top5_indices_df 到 CSV 文件
    top5_values_filename = f"{output_dir}/top5_values_{timestamp}.csv"
    top5_indices_filename = f"{output_dir}/top5_indices_{timestamp}.csv"
    top5_df.to_csv(top5_values_filename, index=False)
    top5_indices_df.to_csv(top5_indices_filename, index=False)

    # 输出文件的完整路径
    top5_values_path = os.path.abspath(top5_values_filename)
    top5_indices_path = os.path.abspath(top5_indices_filename)
    print(f"Top 5 Values CSV saved to: {top5_values_path}")
    print(f"Top 5 Indices CSV saved to: {top5_indices_path}")

    # 返回结果及文件路径
    return  top5_values_path, top5_indices_path,chart_filename

class Moa_processor:
    def __init__(self, test_df):
        self.test_df = test_df
    def Moa_precessor(self):
        test_data = preprocess_data(self.test_df)
        train_targets_scored = pd.read_csv('input/train_targets_scored.csv')
        target_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
        test = test_data[test_data['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
        cat_features = ['cp_time', 'cp_dose']
        num_features = [c for c in self.test_df.columns if self.test_df.dtypes[c] != 'object']
        num_features = [c for c in num_features if c not in cat_features]
        num_features = [c for c in num_features if c not in target_cols]
        test = cate2num(test)
        CFG = {
            "max_grad_norm": 1000,
            "gradient_accumulation_steps": 1,
            "hidden_size": 512,
            "dropout": 0.5,
            "lr": 1e-2,
            "weight_decay": 1e-6,
            "batch_size": 10,
            "epochs": 20,
            "num_targets":206,
            # "total_cate_size": 5,
            # "emb_size": 4,
            "num_features": num_features
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred_file_path = process_data(CFG, test,  num_features, cat_features, device, fold_num=0, seed=42)
        return pred_file_path

class Moa_processor2:
    def __init__(self, test_df):
        self.test_df = test_df
    def Moa_precessor(self):
        test_data = preprocess_data(self.test_df)
        train_targets_scored = pd.read_csv('input/train_targets_scored.csv')
        target_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
        test = test_data[test_data['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
        cat_features = ['cp_time', 'cp_dose']
        num_features = [c for c in self.test_df.columns if self.test_df.dtypes[c] != 'object']
        num_features = [c for c in num_features if c not in cat_features]
        num_features = [c for c in num_features if c not in target_cols]
        test = cate2num(test)
        CFG = {
            "max_grad_norm": 1000,
            "gradient_accumulation_steps": 1,
            "hidden_size": 512,
            "dropout": 0.5,
            "lr": 1e-2,
            "weight_decay": 1e-6,
            "batch_size": 10,
            "epochs": 20,
            "num_targets":206,
            # "total_cate_size": 5,
            # "emb_size": 4,
            "num_features": num_features
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        top_5_path,top_5_indices_path,png_path=process_data2(CFG, test,  num_features, cat_features, device, fold_num=0, seed=42)
        return top_5_path,top_5_indices_path,png_path
import os
import pandas as pd

def test_moa_processor():
    # 定义输入目录
    input_dir = "output_folder/output2.csv"

    test_df = pd.read_csv(input_dir)

    # 初始化 Moa_processor 类
    moa_processor = Moa_processor2(test_df)

    # 调用 Moa_processor 的 Moa_precessor 方法处理数据并保存预测结果
    top_5_path, top_5_indices_path,png_path= moa_processor.Moa_precessor()

    # 打印保存的文件路径
    print(f"Predictions saved to: {top_5_path},{ top_5_indices_path}")

    # 返回保存的文件路径
    return top_5_path, top_5_indices_path,png_path

# 调用测试函数
if __name__ == "__main__":
    test_moa_processor()








