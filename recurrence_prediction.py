from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from data_pre_process import pre_process
from model import train_model, eval_model
from matplotlib import pyplot as plt
import lightgbm as lgb
import sys

sys.argv.append(".\data.xls")

Label_Index = -1
Drop_Index = [38, 39, 40]
Cat_Index = [0, 6, 7, 10, 11, 35, 37]

if __name__ == '__main__':
    
    data = pre_process(sys.argv[1], "median")
    C_NAMES = data.columns
    
    label = data[C_NAMES[Label_Index]]
    
    for index in Drop_Index:
        data.drop([C_NAMES[index]], axis=1, inplace=True)
    for index in Cat_Index:
        pass
        # data[C_NAMES[index]] = data[C_NAMES[index]].astype('category')
    
    data_model, data_test, label_model, label_test = train_test_split(data, label, test_size=0.2, random_state=2020)

    # 五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=2020)
    for train_index, valid_index in kf.split(data_model):
        # 生成训练集和验证集
        data_train = data_model.copy()
        data_valid = data_model.copy()
        data_train.drop(data_train.index[valid_index], inplace=True)
        data_valid.drop(data_valid.index[train_index], inplace=True)

        # 得到label
        label_train = label_model.drop(label_model.index[valid_index])
        label_valid = label_model.drop(label_model.index[train_index])

        gbm_model, evals_result = train_model(data_train, label_train, data_valid, label_valid)
        score = eval_model(data_valid, label_valid)

    # 全训练集建模
    print("全训练集建模")
    gbm_model, evals_result = train_model(data_model, label_model, data_test, label_test)
    # 模型特征重要性画图
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    lgb.plot_importance(gbm_model)
    plt.show()
    # 测试集评测
    eval_model(data_test, label_test)