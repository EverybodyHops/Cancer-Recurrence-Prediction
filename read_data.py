import pandas as pd
import os, sys

def read_raw_data(file_name, sheet_name="3月4日"):
    _data = pd.read_excel(file_name, sheet_name=sheet_name)
    _data.drop(len(_data[_data.columns[0]]) - 1, inplace=True)
    return _data


if __name__ == "__main__":
    data = read_raw_data(sys.argv[1])
    print(data)
    