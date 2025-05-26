import pandas as pd
import os

import pandas as pd
from sklearn.decomposition import PCA

import numpy as np

def cate2num(df):
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72: 2})
    df['cp_dose'] = df['cp_dose'].map({'D1': 3, 'D2': 4})
    return df
def batch_pca(data, n_components, batch_size=100):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca_data = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        pca_data.append(pca.fit_transform(batch))
    return np.vstack(pca_data)
from sklearn.decomposition import PCA


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

    # 提取数值特征

# 测试代码
if __name__ == "__main__":
    # 定义输入文件路径
    input_dir = "output_folder"
    test_file_path = os.path.join(input_dir, "output1.csv")

    # 读取测试集 CSV 文件
    test_df = pd.read_csv(test_file_path)

    # 调用 preprocess_data 函数进行数据预处理
    top_5_value,top_5_indices= preprocess_data(test_df)

    # 打印处理后的数据形状
    print("Shape of processed test features:", top_5_value)