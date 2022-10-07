"""
差分进化算法差分变异
"""
import random
import numpy as np

def mutate(pops, F, lb, rb):
    # 获取种群的规模 (nPop) 以及种群维度 (nChr)
    nPop, nChr = pops.shape
    # 初始化变异种群
    mutantPops = np.zeros((nPop, nChr))
    indices = np.arange(nPop).tolist()
    for i in range(nPop):
        # 随机抽取 3 个种群的下标
        rs = random.sample(indices, 3)
        # 通过下面的方法：由随机抽取的三个种群，获得变异种群
        # !!! 下面的计算方法跟论文中的不一样，为什么？
        mutantPops[i] = pops[rs[0]] + F * (pops[rs[1]] - pops[rs[2]])
        # 检查是否越界
        for j in range(nChr):
            if mutantPops[i, j] < lb:
                mutantPops[i, j] = lb
            if mutantPops[i, j] > rb:
                mutantPops[i, j] = rb

    return mutantPops       # 变异的种群的数据形状为： nPop x nChr，即与原始种群形状一致