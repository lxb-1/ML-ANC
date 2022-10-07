"""
种群或个体的适应度
"""

import numpy as np
from function import *

def fitness(pops, func):
    """群体或个体适应度计算函数

    Args:
        pops (_type_): 种群
        func (_type_): 目标函数

    Returns:
        fits (_type_) : 适应度，其形状为 种群规模 (nPop) * 目标函数个数 (nF)
    """
    # 如果是 1 维则需要转换为 2 维
    if pops.ndim == 1:
        pops = pops.reshape(1, len(pops))
    # 得到种群的规模 nPop
    nPop = pops.shape[0]
    # 计算种群 / 个体的适应度
    fits = np.array([func(pops[i]) for i in range(nPop)])

    return fits

if __name__ == "__main__":
    pops = np.array([[-0.57735, -0.57735, -0.57735, -0.57735], [0.2, 0.2, 0.2, 0.2]])
    fits = fitness(pops, function)
    print(fits)