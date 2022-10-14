'''
@File    :   MODE.py
@Time    :   2020/11/04
@Author  :   Yu Li
@describe:   基于快速非支配排序算法的多目标差分进化算法主程序 
'''

from mutate import *
from crossover import *
from select1 import *
from nonDominationSort import crowdingDistanceSort, nonDominationSort
from fitness import fitness
from initPop import initPop
import numpy as np

def MODE(nIter, nChr, nPop, F, Cr, func, lb, rb):
    """多目标差分进化算法主程序

    Args:
        nIter (_type_): 迭代次数
        nChr (_type_): ?? 种群的维度
        nPop (_type_): 种群规模，对应于论文中的 `N`
        F (_type_): 缩放因子，对应于论文中的 `F`
        Cr (_type_): 交叉概率，对应于论文中的 `R`
        func (_type_): 优化函数
        lb (_type_): 自变量下界
        rb (_type_): 子变量上界

    Returns:
        paretoPops (_type_): Pareto 解集
        paretoFits (_type_): 对应的适应度
    """

    # 使用 initPop 方法与 fitness 方法生成初始种群 parPops 及其适应度 parFits
    parPops = initPop(nChr, nPop, lb, rb)
    parFits = fitness(parPops, func)

    # 开始迭代
    iter = 1
    while iter <= nIter:
        # 现实进度条
        print("【进度】【{0:20s}】【正在进行{1}代...】【共{2}代】".\
            format('▋'*int(iter/nIter*20), iter, nIter), end='\r')

        # 产生变异向量
        mutantPops = mutate(parPops, F, lb, rb)
        # 产生实验向量
        trialPops = crossover(parPops, mutantPops, Cr)
        # 重新计算适应度
        trialFits = fitness(trialPops, func)

        # 合成新的物种。？？？？这里我有个问题没有搞明白，新合成的物种的种群规模的变化？？？？？？
        pops = np.concatenate((parPops, trialPops), axis=0)
        fits = np.concatenate((parFits, trialFits), axis=0)
        ranks = nonDominationSort(pops, fits)               # 非支配排序
        distances = crowdingDistanceSort(pops, fits, ranks) # 计算拥挤度

        # 种群选择
        parPops, parFits = select1(nPop, pops, fits, ranks, distances)

        iter += 1
    print("\n")
    # 获取等级为 0，即实际求解得到的 Pareto 前沿
    paretoPops = pops[ranks==0]
    paretoFits = fits[ranks==0]

    return paretoPops, paretoFits