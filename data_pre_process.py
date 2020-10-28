from read_data import read_raw_data
import pandas as pd
import numpy as np
import sys

NEW_FILE = "new_data.xls"

SEX_INDEX = 1
AGE_INDEX = 2
CHILD_INDEX = 9
SYNDORME_INDEX = 41
SERIOUS_SYNDORME_INDEX = 42

def pre_process(file_name):
    df = read_raw_data(file_name)
    data_cleaning(df)
    return df

def data_cleaning(df):
    c_names = df.columns
    
    # 将性别转为数字 原来为字符串
    c_name_index = SEX_INDEX
    s = df[c_names[c_name_index]]
    t = np.zeros(len(s))
    for i, sex in enumerate(s):
        if sex != "0":
            t[i] = 1.0
    df[c_names[c_name_index]] = pd.Series(t)

    # 将年龄转为数字 原来为字符串
    c_name_index = AGE_INDEX
    s = df[c_names[c_name_index]]
    t = np.zeros(len(s))
    for i, age in enumerate(s):
        try:
            t[i] = float(age)
        except ValueError:
            t[i] = np.nan
    df[c_names[c_name_index]] = pd.Series(t)
    
    # 处理Child分级 A为0 B为1
    c_name_index = CHILD_INDEX
    s = df[c_names[c_name_index]]
    t = np.zeros(len(s))
    for i, child in enumerate(s):
        if child == "B":
            t[i] = 1.0
    df[c_names[c_name_index]] = pd.Series(t)

    # 处理严重并发症 和 并发症合为一列
    # 无并发症 0 有并发症 1 并发症严重 2
    
    for i, if_have in enumerate(df[c_names[SYNDORME_INDEX]]):
        if if_have == 1.0:
            if df[c_names[SERIOUS_SYNDORME_INDEX]][i] != "0":
                df.iloc[i, SYNDORME_INDEX] = 2.0
    s = df[c_names[SYNDORME_INDEX]]
    df.drop([c_names[SERIOUS_SYNDORME_INDEX]], axis=1, inplace=True)

if __name__ == "__main__":
    data = pre_process(sys.argv[1])
    data.to_excel(NEW_FILE)
    # n_df = pd.read_excel(NEW_FILE, dtype="float64")
    # print(n_df)