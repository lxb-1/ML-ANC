"""
种群或个体的适应度
"""

import numpy as np
from function import *

def fitness(pops, func):
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