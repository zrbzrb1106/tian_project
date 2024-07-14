
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def load_data_to_libsvm(path):
    df = pd.read_excel(path)
    print(df.columns)
    num_cols = len(df.columns)
    print(num_cols)
    for index, row in df.iterrows():
        pass

if __name__ == "__main__":

    # 加载数据到 Pandas DataFrame 中（假设数据已经在 Excel 文件中）
    df = pd.read_excel('医学数据0704.xlsx').dropna(axis=1, how='all').fillna(0)
    print(df.columns)

    # 假设 df 中最后一列是目标变量，其余列是特征
    X = df.iloc[:, 2:-3]  # 特征
    y1 = df.iloc[:, -3]
    y2 = df.iloc[:, -2]

    y1 = y1.astype("int")
    y2 = y2.astype("int")
    y = pd.concat([y1, y2], axis=1)
    X["手术"] = X["手术"].astype("category")
    X["睡眠药"] = X["睡眠药"].astype("category")
    X["其他疾病"] = X["其他疾病"].astype("category")
    print(y)

    # 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    params = {
        'objective': 'binary:logistic',  # 二分类逻辑回归
        'eval_metric': 'logloss',         # 使用对数损失作为评估指标
        'max_depth': 10,                   # 树的最大深度
        'eta': 0.1,                       # 学习率
        'gamma': 0.3                 # 控制节点分裂的最小损失减少量
    }

    model = xgb.train(params, dtrain, 5)

    # 在测试集上做出预测
    y_pred_train = model.predict(dtrain)
    y_pred = model.predict(dtest)

    # 计算AUC
    res_auc_train = roc_auc_score(y_train, y_pred_train, average=None)
    print(res_auc_train)
    res_auc_test = roc_auc_score(y_test, y_pred, average=None)
    print(res_auc_test)

    # 导出模型
    model.save_model('xgboost_model.xgb')
    