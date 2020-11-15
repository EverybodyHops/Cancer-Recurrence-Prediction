from sklearn.model_selection import KFold
from data_pre_process import pre_process
import sys
from model import train_model

if sys.argv:
    sys.argv.append(".\data.xls")

Label_Index = -1
Drop_Index = [39,40]

if __name__ == '__main__':
    data = pre_process(sys.argv[1])
    #   数据分成五份
    kf = KFold(n_splits=5, shuffle=True, random_state=2020)
    for test_index, train_index in kf.split(data):
        #   生成训练集和测试集
        data_train = data.copy()
        data_test = data.copy()
        data_train.drop(data_train.index[train_index], inplace=True)
        data_test.drop(data_test.index[test_index], inplace=True)

        C_NAMES_train = data_train.columns
        C_NAMES_test = data_test.columns
        #   得到label
        data_train_label = data_train[C_NAMES_train[Label_Index]]
        data_test_label = data_test[C_NAMES_test[Label_Index]]
        #   删除label列和随机访问时间列
        for index in Drop_Index:
            data_train.drop([C_NAMES_train[index]], axis=1, inplace=True)
            data_test.drop([C_NAMES_test[index]], axis=1, inplace=True)

        gbm_model, evals_result = train_model(data_train, data_train_label, data_test, data_test_label)


