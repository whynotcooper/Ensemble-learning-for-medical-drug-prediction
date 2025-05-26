import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

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








class TrainDataset(Dataset):
    def __init__(self, df, num_features, cat_features, labels):
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values
        self.labels = labels

    def __len__(self):
        return len(self.cont_values)

    def __getitem__(self, idx):
        cont_x = torch.FloatTensor(self.cont_values[idx])
        cate_x = torch.LongTensor(self.cate_values[idx])
        label = torch.tensor(self.labels[idx]).float()
        return cont_x, cate_x, label


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

cat_features = ['cp_time', 'cp_dose']
num_features = [c for c in train.columns if train.dtypes[c] != 'object']
num_features = [c for c in num_features if c not in cat_features]
num_features = [c for c in num_features if c not in target_cols]
target = train[target_cols].values

def cate2num(df):
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72: 2})
    df['cp_dose'] = df['cp_dose'].map({'D1': 3, 'D2': 4})
    return df

train = cate2num(train)
test = cate2num(test)
import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: 模型的输出，预测的概率
        # targets: 真实标签
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv1dBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()  # 修改为非 inplace 操作

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class FSPPConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(FSPPConv1d, self).__init__()
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()  # 修改为非 inplace 操作
            ))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()  # 修改为非 inplace 操作
        )
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()  # 修改为非 inplace 操作
        )
        self.conv_out = nn.Conv1d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        conv1x1_output = self.conv1x1(x)
        atrous_outputs = [conv(x) for conv in self.atrous_convs]
        outputs = [conv1x1_output] + atrous_outputs
        outputs = torch.cat(outputs, dim=1)
        return self.conv_out(outputs)


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size=1)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv3 = Conv1dBlock(out_channels, out_channels * self.expansion, kernel_size=1)
        self.relu = nn.ReLU()  # 修改为非 inplace 操作
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity  # 修改为非 inplace 操作
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.sigmoid= nn.Sigmoid()
        self.relu = nn.ReLU()  # 修改为非 inplace 操作
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d((1,))
        self.drop= nn.Dropout(0.2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.ASPPF = FSPPConv1d(512, 512, atrous_rates=[3, 3, 3, 3, 3])

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x=self.sigmoid(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x=self.drop(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.drop(x)
        return x


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3, 2])

class MLP1(nn.Module):
    def __init__(self):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(28, 28)
        self.fc2 = nn.Linear(28, 1)
        self.fc3 = nn.Linear(1024, 206)
        self.relu = nn.ELU()

    def forward(self, x):
        x = x.view(x.size(0), -1, 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1, 1)  # 转换为 (1, 4096, 1)
        x = x.view(x.size(0), 1, -1)
        x = self.fc3(x)
        return x


def attndot(Q, K):
    batch = Q.shape[0]
    Q_row = Q.shape[2]
    K_row = K.shape[3]
    Q_row_sum = torch.sum(torch.pow(Q, 2), dim=3)
    K_row_sum = torch.sum(torch.pow(K, 2), dim=2)
    Q_row_sum_expanded = Q_row_sum.unsqueeze(2).expand(-1, -1, -1, K_row)

    K_row_sum_expanded = K_row_sum.unsqueeze(1).expand(-1, -1, Q_row, -1)

    result_mul = torch.mul(Q_row_sum_expanded, K_row_sum_expanded)
    result_mul = torch.sqrt(result_mul)
    result_add = torch.add(Q_row_sum_expanded, K_row_sum_expanded)
    result = torch.div(result_mul, result_add)
    return result


def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.W1(x))
        x = self.W2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义权重矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        # 线性变换
        Q = self.W_Q(Q)
        print("Q shape:", Q.shape)
        K = self.W_K(K)
        print("K shape:", K.shape)
        V = self.W_V(V)
        print("V shape:", V.shape)
        # 分割成多个头
        Q = Q.view(Q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(K.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(V.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        print("Q shape after split:", Q.shape)
        # 计算注意力得分
        scores = Q.matmul(K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        print("scores shape:", scores.shape)
        attn = F.softmax(scores, dim=-1)

        # 计算注意力输出
        output = attn.matmul(V)

        # 拼接多个头的输出
        output = output.transpose(1, 2).contiguous().view(Q.size(0), -1, self.d_model)

        # 线性变换
        output = self.W_O(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.mha.forward(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.layernorm1(x)
        ff_output = self.ff.forward(x)
        x = x + self.dropout(ff_output)
        x = self.layernorm2(x)
        return x
class MOA(nn.Module):
    def __init__(self):
        super(MOA, self).__init__()
        self.resnet = ResNet50()
        self.ASPP=FSPPConv1d(1024, 1024, atrous_rates=[3,3,3,3,3])
        self.mlp = MLP1()
        self.encoder=EncoderLayer(1024,8,1024)
    def forward(self, x,target):
        x=x.view(x.shape[0],1,x.shape[1])
        x = self.resnet(x)
        x = self.ASPP(x)
        x=x.view(x.shape[0],-1,1024)
        x=self.encoder(x)
        x = self.mlp(x)
        return x
class CFG:
    max_grad_norm=1000
    gradient_accumulation_steps=1
    hidden_size=512
    dropout=0.5
    lr=1e-2
    weight_decay=1e-6
    batch_size=1
    epochs=20
    #total_cate_size=5
    #emb_size=4
    num_features=num_features
    cat_features=cat_features
    target_cols=target_cols


def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    losses = AverageMeter()

    model.train()

    for step, (cont_x, cate_x, y) in enumerate(train_loader):

        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        pred = model(cont_x, cate_x)
        pred=pred.view(pred.shape[0],-1)
        loss = nn.BCEWithLogitsLoss()(pred, y)
        losses.update(loss.item(), batch_size)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        loss.backward(retain_graph=True)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

    return losses.avg


def validate_fn(valid_loader, model, device):
    losses = AverageMeter()

    model.eval()
    val_preds = []

    for step, (cont_x, cate_x, y) in enumerate(valid_loader):

        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        with torch.no_grad():
            pred = model(cont_x, cate_x)
        pred = pred.view(pred.shape[0], -1)
        loss = nn.BCEWithLogitsLoss()(pred, y)
        losses.update(loss.item(), batch_size)

        val_preds.append(pred.sigmoid().detach().cpu().numpy())

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

    val_preds = np.concatenate(val_preds)

    return losses.avg, val_preds


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_single_nn(cfg, train, test, folds, num_features, cat_features, target, device, fold_num=0, seed=42):
    # Set seed
    logger.info(f'Set seed {seed}')
    seed_everything(seed=seed)

    # loader
    trn_idx = folds[folds['fold'] != fold_num].index
    val_idx = folds[folds['fold'] == fold_num].index
    train_folds = train.loc[trn_idx].reset_index(drop=True)
    valid_folds = train.loc[val_idx].reset_index(drop=True)
    train_target = target[trn_idx]
    valid_target = target[val_idx]
    train_dataset = TrainDataset(train_folds, num_features, cat_features, train_target)
    valid_dataset = TrainDataset(valid_folds, num_features, cat_features, valid_target)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=False)

    # model
    model = MOA()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=cfg.epochs, steps_per_epoch=len(train_loader))

    # log
    log_df = pd.DataFrame(columns=(['EPOCH'] + ['TRAIN_LOSS'] + ['VALID_LOSS']))

    # train & validate
    best_loss = np.inf
    for epoch in range(cfg.epochs):
        train_loss = train_fn(train_loader, model, optimizer, epoch, scheduler, device)
        valid_loss, val_preds = validate_fn(valid_loader, model, device)
        log_row = {'EPOCH': epoch,
                   'TRAIN_LOSS': train_loss,
                   'VALID_LOSS': valid_loss,
                   }
        log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)
        # logger.info(log_df.tail(1))
        if valid_loss < best_loss:
            logger.info(f'epoch{epoch} save best model... {valid_loss}')
            best_loss = valid_loss
            oof = np.zeros((len(train), len(cfg.target_cols)))
            oof[val_idx] = val_preds
            torch.save(model.state_dict(), f"fold{fold_num}_seed{seed}.pth")

    # predictions
    test_dataset = TestDataset(test, num_features, cat_features)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    model = MOA()
    model.load_state_dict(torch.load(f"fold{fold_num}_seed{seed}.pth"))
    model.to(device)
    predictions = inference_fn(test_loader, model, device)

    # del
    torch.cuda.empty_cache()

    return oof, predictions


def run_kfold_nn(cfg, train, test, folds, num_features, cat_features, target, device, n_fold=5, seed=42):
    oof = np.zeros((len(train), len(cfg.target_cols)))
    predictions = np.zeros((len(test), len(cfg.target_cols)))

    for _fold in range(n_fold):
        logger.info("Fold {}".format(_fold))
        _oof, _predictions = run_single_nn(cfg,
                                           train,
                                           test,
                                           folds,
                                           num_features,
                                           cat_features,
                                           target,
                                           device,
                                           fold_num=_fold,
                                           seed=seed)
        oof += _oof
        predictions += _predictions / n_fold

    score = 0
    for i in range(target.shape[1]):
        _score = log_loss(target[:, i], oof[:, i])
        score += _score / target.shape[1]
    logger.info(f"CV score: {score}")




    return oof, predictions

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
    return  top5_values_path, top5_indices_path
