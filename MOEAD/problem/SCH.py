'''
求解问题部分
'''

import numpy as np

Dimention = 30          # 函数维度（目标维度不一致的自行编写目标函数）
Func_num = 2            # 目标函数个数
Bound = [-1000, 1000]   # 函数变量边界

def Func(X):
    f1 = F1(X)
    f2 = F2(X)
    return [f1, f2]

def F1(X):
    return X[0] ** 2

def F2(X):
    return (X[0] - 2) ** 2