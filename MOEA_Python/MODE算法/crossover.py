"""
杂交算子
"""

from random import choices
import numpy as np

def crossover(pops, mutantPops, Cr):
    """杂交算子

    Args:
        pops (_type_): 种群
        mutantPops (_type_): 突变种群
        Cr (_type_): 交叉概率因子，对应论文中的 R

    Returns:
        trailPops (_type_): 实验种群
    """

    # 获取种群的规模 (nPop) 及种群维度 (nChr)
    nPop, nChr = pops.shape

    # 选择变异向量的位置 1
    choiMuPops1 = np.random.rand(nPop, nChr) < Cr
    # 选择变异向量的位置 2
    choiMuPops2 = np.random.randint(0, nPop, (nPop, nChr)) == \
        np.tile(np.arange(nChr), (nPop, 1))

    choiMuPops = choiMuPops1 | choiMuPops2
    choiPops = ~ choiMuPops
    trailPops = mutantPops * choiMuPops + pops * choiPops

    return trailPops