'''
求解问题部分
'''

import numpy as np

Dimention = 30      # 函数的维度（目标维度不一致的自行编写目标函数）
Func_num = 2        # 目标函数的个数
Bound = [0, 1]      # 函数边界

def Func(X):
    if X.shape[0] < 2:
        return -1
    f1 = F1(X)
    ag = g(X)
    f2 = F2(ag, X)
    return [f1, f2]

def F1(X):
    return X[0]

def F2(gx, X):
    x = X[0]
    f2 = gx * (1 - np.square(x / gx))
    return f2

def g(X):
    g = 1 + 9 * (np.sum(X[1:], axis=0) / (X.shape[0] - 1))
    return g