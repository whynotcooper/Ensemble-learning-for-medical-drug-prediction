import pandas as pd
import os
from process_data import test_moa_processor
from process_data import Moa_processor2
def process_excel_data(df):
    """示例数据处理算法"""

    # 返回结构化结果
    return "{result: success}"

import pandas as pd
import os

def process_excel_data2(file_path):
    # 获取文件扩展名
    file_extension = os.path.splitext(file_path)[1].lower()

    # 根据文件扩展名选择处理方式
    if file_extension == '.csv':
        # 读取 CSV 文件
        df = pd.read_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        # 根据 Excel 文件格式选择引擎
        engine = 'openpyxl' if file_extension == '.xlsx' else 'xlrd'
        # 读取 Excel 文件
        df = pd.read_excel(file_path, engine=engine)
    else:
        raise ValueError("Unsupported file format. Only .csv, .xls, and .xlsx are supported.")
    processor=Moa_processor2(df)
    result_file_path,result_indices,result_png=processor.Moa_precessor()
    return result_file_path,result_indices,result_png
def process_excel_data3(file_path):
    return test_moa_processor(file_path)
