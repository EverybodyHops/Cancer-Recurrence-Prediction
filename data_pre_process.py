from read_data import read_raw_data
import pandas as pd
import numpy as np
import sys

C_NAMES = None
NEW_FILE = "new_data.xls"


SEX_INDEX = 1
AGE_INDEX = 2
CHILD_INDEX = 9
SYNDORME_INDEX = 41
SERIOUS_SYNDORME_INDEX = 42

ABC_INDEX = [3, 4, 5]

USELESS_INDEX = []

def pre_process(file_name):
    global C_NAMES
    df = read_raw_data(file_name)
    C_NAMES = df.columns
    data_cleaning(df)
    data_padding(df, normal_method="median")
    return df

# 进行数据的格式转换和某些列的删除
def data_cleaning(df):
    # 将性别转为数字 原来为字符串
    c_name_index = SEX_INDEX
    s = df[C_NAMES[c_name_index]]
    t = np.zeros(len(s))
    for i, sex in enumerate(s):
        if sex != "0":
            t[i] = 1.0
    df[C_NAMES[c_name_index]] = pd.Series(t)

    # 将年龄转为数字 原来为字符串
    c_name_index = AGE_INDEX
    s = df[C_NAMES[c_name_index]]
    t = np.zeros(len(s))
    for i, age in enumerate(s):
        try:
            t[i] = float(age)
        except ValueError:
            t[i] = np.nan
    df[C_NAMES[c_name_index]] = pd.Series(t)
    
    # 处理Child分级 A为0 B为1
    c_name_index = CHILD_INDEX
    s = df[C_NAMES[c_name_index]]
    t = np.zeros(len(s))
    for i, child in enumerate(s):
        if child == "B":
            t[i] = 1.0
    df[C_NAMES[c_name_index]] = pd.Series(t)

    # 处理严重并发症 和 并发症合为一列
    # 无并发症 0 有并发症 1 并发症严重 2
    
    for i, if_have in enumerate(df[C_NAMES[SYNDORME_INDEX]]):
        if if_have == 1.0:
            if df[C_NAMES[SERIOUS_SYNDORME_INDEX]][i] != "0":
                df.iloc[i, SYNDORME_INDEX] = 2.0
    s = df[C_NAMES[SYNDORME_INDEX]]
    df.drop([C_NAMES[SERIOUS_SYNDORME_INDEX]], axis=1, inplace=True)

# 填充缺失数据
def data_padding(df, normal_method="mean"):
    method_dic = {
        "mean": df.mean,
        "median": df.median
    }
    normal_padding = method_dic.get(normal_method, df.mean)()

    # 填充abc列
    df[C_NAMES[ABC_INDEX[0]]].fillna(normal_padding[C_NAMES[ABC_INDEX[0]]], inplace=True)
    for i, v in enumerate(df[C_NAMES[ABC_INDEX[0]]]):
        if pd.isna(df.iloc[i, ABC_INDEX[1]]):
            df.iloc[i, ABC_INDEX[1]] = v
            df.iloc[i, ABC_INDEX[2]] = v
    for i, v in enumerate(df[C_NAMES[ABC_INDEX[1]]]):
        if pd.isna(df.iloc[i, ABC_INDEX[2]]):
            df.iloc[i, ABC_INDEX[2]] = v

if __name__ == "__main__":
    data = pre_process(sys.argv[1])
    data.to_excel(NEW_FILE)
    # n_df = pd.read_excel(NEW_FILE, dtype="float64")
    # print(n_df)
