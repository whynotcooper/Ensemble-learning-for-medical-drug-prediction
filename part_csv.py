import os
import pandas as pd

# 定义文件夹路径
folder_name = "output_folder"

# 创建文件夹（如果不存在）
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 定义输入和输出文件路径
input_file = 'input/test_features.csv'
output_file = os.path.join(folder_name, 'output2.csv')

# 读取 CSV 文件
df = pd.read_csv(input_file)

# 截取前 10 行数据
first_10_rows = df.head(2)


# 将截取的数据保存到新文件夹中的新 CSV 文件
first_10_rows.to_csv(output_file, index=False)

print(f"前 10 行数据已保存到 {output_file}")