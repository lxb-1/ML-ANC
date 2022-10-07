"""
选择算子
"""

import random
import numpy as np

def select1(pool, pops, fits, ranks, distances):
    # 一对一锦标赛选择
    # pool : 新生成的总群大小

    nPop, nChr = pops.shape     # 获取种群规模 nPop 与种群维度 nChr
    nF = fits.shape[1]          # 获取目标函数的个数

    newPops = np.zeros((pool, nChr))    # 初始化新的种群矩阵，其种群规模为 pool，种群维度为 nChr
    newFits = np.zeros((pool, nF))      # 初始化新的适应度矩阵，其形状为 pool * nF

    indices = np.arange(nPop).tolist()  # 下标。由于 nPop 为 ndarray 格式的数据，因此需要使用 tolist() 方法将其转换为列表
    i = 0
    while i < pool:
        idx1, idx2 = random.sample(indices, 2)      # 随机挑选两个个体
        idx = compare(idx1, idx2, ranks, distances) # 返回最优的 idx
        # 更新
        newPops[i] = pops[idx]
        newFits[i] = fits[idx]
        i += 1
    return newPops, newFits

def compare(idx1, idx2, ranks, distances):
    # 返回值：最优的 idx
    if ranks[idx1] < ranks[idx2]:
        idx = idx1
    elif ranks[idx1] > ranks[idx2]:
        idx = idx2
    else:
        # 等级相同，则比较拥挤度，拥挤度越大越好 ！！！！！！
        if distances[idx1] <= distances[idx2]:
            idx = idx2
        else:
            idx = idx1
    return idx