import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

# 参数设置
parameters = {
                  'objective': 'binary',
                  'verbose': -1,
                  
                  'metric': 'auc',         # 测试集使用的指标
                  
                  'max_depth': 4,          # 树的最大深度
                  'num_leaves': 5,         # 叶子节点数
                  'min_data_in_leaf': 20,  # 叶子节点上的数据的最小数量
                  'learning_rate': 0.01,   # 收敛速度
                  # 'feature_fraction': 0.6, # 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征.
                  # 'bagging_fraction': 0.6, # 类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
                  # 'bagging_freq': 6,       # bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
                  'lambda_l1': 0.08,        # L1正则化
                  }

def train_model(data_train, label_train, data_valid, label_valid):

    lgb_train = lgb.Dataset(data_train, label_train)
    lgb_valid = lgb.Dataset(data_valid, label_valid, reference=lgb_train)
    
    print('Starting training...')
    # 模型训练
    evals_result = {}  # 记录训练结果所用
    gbm_model = lgb.train(parameters,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_valid],
                    num_boost_round=200,  # 提升迭代的次数
                    early_stopping_rounds=50,
                    evals_result=evals_result,
                    verbose_eval=50
                    )
    
    # 模型特征重要性画图
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    lgb.plot_importance(gbm_model)
    plt.show()
    
    print('Saving model...')
    # 模型保存
    gbm_model.save_model('model.txt')
    print('Done!')
    
    return gbm_model, evals_result

def eval_model(data_test, label_test):
    
    # 模型加载
    gbm_model = lgb.Booster(model_file='model.txt')
    
    print('Starting predicting...')
    # 模型预测
    label_test_pred = gbm_model.predict(data_test, num_iteration=gbm_model.best_iteration)
    
    # 模型评估
    score = roc_auc_score(label_test, label_test_pred)
    
    print(score)
    print('Done!')
    
    return score