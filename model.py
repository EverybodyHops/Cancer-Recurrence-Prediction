import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# 参数设置
parameters = {
                  'task': 'train',
                  'max_depth': 15,
                  'boosting_type': 'gbdt',
                  'num_leaves': 20,        # 叶子节点数
                  'n_estimators': 50,
                  'objective': 'binary',
                  'metric': 'auc',
                  'learning_rate': 0.2,
                  'feature_fraction': 0.7, #小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征.
                  'bagging_fraction': 1,   #类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
                  'bagging_freq': 3,       #bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
                  'lambda_l1': 0.5,
                  'lambda_l2': 0,
                  'cat_smooth': 10,        #用于分类特征,这可以降低噪声在分类特征中的影响, 尤其是对数据很少的类别
                  'is_unbalance': False,   #适合二分类。这里如果设置为True，评估结果降低3个点
                  'verbose': 0
                  }

def train_model(data_train, data_train_label, data_test, data_test_label):

    lgb_train = lgb.Dataset(data_train, data_train_label)
    lgb_test = lgb.Dataset(data_test, data_test_label, reference=lgb_train)
    print('Starting training...')
    # 模型训练
    evals_result = {}  # 记录训练结果所用
    gbm_model = lgb.train(parameters,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    num_boost_round=50,  # 提升迭代的次数
                    early_stopping_rounds=5,
                    evals_result=evals_result,
                    verbose_eval=10
                    )
    print('Saving model...')
    # 模型保存
    gbm_model.save_model('model.txt')
    # 模型加载
    gbm_model = lgb.Booster(model_file='model.txt')
    print('Starting predicting...')
    # 模型预测
    data_test_pred = gbm_model.predict(data_test, num_iteration=gbm_model.best_iteration)
    # 模型评估
    score = roc_auc_score(data_test_label, data_test_pred)
    print(score)
    return gbm_model, evals_result