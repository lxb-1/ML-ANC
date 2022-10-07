"""
快速非支配排序
"""
import random
import numpy as np

def nonDominationSort(pops, fits):
    """快速非支配排序算法

    Args:
        pops (_type_): 种群， nPop * nChr 的数组，其中 nPop 为种群规模， nChr 为种群维度
        fits (_type_): 适应度， nPop * nF 的数组，其中 nF 为目标函数的个数

    Returns:
        rank (_type_): 每个个体对应的等级，一维数组。
    """
    nPop = pops.shape[0]
    # 通过适应度返回值 fits 的第二个维度得到目标函数的个数
    nF = fits.shape[1]
    # 初始化每个个体 (种群？) 的等级
    ranks = np.zeros(nPop, dtype=np.int32)
    # 初始化每个个体 p 被支配的个数，初始化为零
    nPs = np.zeros(nPop)
    # 每个个体支配的解的集合，后续将索引放入
    sPs = []
    for i in range(nPop):
        iSet = []   # 解 i 的支配解集
        for j in range(nPop):
            if i == j:  # 如果是个体本身与自己比较，则进入下一次循环
                continue
            isDom1 = fits[i] <= fits[j]
            isDom2 = fits[i] <  fits[j]
            # 判断解i 是否支配解 j
            if sum(isDom1) == nF and sum(isDom2) >= 1:
                iSet.append(j)
            # 判断解j 是否支配 i
            if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                nPs[i] += 1
        # 添加 i 支配的解的索引到 sPs中
        sPs.append(iSet)
    """!!!!当前等级为 0，等级越低越好!!!!"""
    r = 0
    indices = np.arange(nPop)
    while sum(nPs==0) != 0:
        rIdices = indices[nPs==0]   # 当前被支配数为 0 的索引
        ranks[rIdices] = r
        for rIdx in rIdices:
            iSet = sPs[rIdx]
            nPs[iSet] -= 1
        nPs[rIdices] = -1   # 当前等级的被支配数设置为负数
        r += 1

    return ranks

# 拥挤度排序算法
def crowdingDistanceSort(pops, fits, ranks):
    """拥挤度排序算法

    Args:
        pops (_type_): 种群，其形状为 nPop * nChr 的数组，其中 nPop 为种群规模； nChr 为种群的维度
        fits (_type_): 适应度矩阵，其形状为 nPop * nF 数组，其中 nF 为目标函数的个数
        ranks (_type_): 每个个体对应的等级的一维数组

    Returns:
        dis (_type_): 每个个体的拥挤度的一维数组
    """
    # 种群规模
    nPop = pops.shape[0]
    # 目标函数个数
    nF = fits.shape[1]
    # 拥挤度初始化
    dis = np.zeros(nPop)
    # 最大等级
    nR = ranks.max()
    indices = np.arange(nPop)
    for r in range(nR + 1):
        # 当前等级种群的索引
        rIdices = indices[ranks == r]
        # 当前等级的种群
        rPops = pops[ranks == r]
        # 当前等级种群的适应度
        rFits = fits[ranks == r]

        # 对纵向排序的索引
        rSortIdices = np.argsort(rFits, axis=0)
        # 得到纵向适应度排序
        rSortFits = np.sort(rFits, axis=0)

        fMax = np.max(rFits, axis=0)    # 得到 rFits 纵向上每个维度的最大值
        fMin = np.min(rFits, axis=0)    # 得到 rFits 纵向上每个维度的最小值

        # 得到当前等级的个体数目
        n = len(rIdices)
        for i in range(nF):     # 对 xx 在纵向进行遍历
            orIdices = rIdices[rSortIdices[:,i]]    # 当前操作元素的原始位置
            j = 1
            while n > 2 and j < n - 1:
                if fMax[i] != fMin[i]:
                    dis[orIdices[j]] += (rSortFits[j+1,i] - rSortFits[j-1,i]) / \
                        (fMax[i] - fMin[i])
                else:
                    dis[orIdices[j]] = np.inf
                j += 1
            dis[orIdices[0]] = np.inf
            dis[orIdices[n-1]] = np.inf

    return dis

# 测试非支配排序算法的正确性
if __name__ == "__main__":
    y1 = np.arange(1, 5).reshape(4, 1)
    y2 = 5 - y1
    fit1 = np.concatenate((y1,y2),axis=1)
    y3 = 6 - y1
    fit2 = np.concatenate((y1,y3),axis=1)
    y4 = 7 - y1
    fit3 = np.concatenate((y1,y4),axis=1)
    fit3 = fit3[:2]
    fits = np.concatenate((fit1,fit2,fit3), axis=0)
    pops = np.arange(fits.shape[0]).reshape(fits.shape[0],1)

    # 打乱数组
    indices = np.arange(fits.shape[0])
    random.shuffle(indices)
    fits = fits[indices]
    pops = pops[indices]
    print(indices)

    # 首先测试非支配排序算法
    ranks = nonDominationSort(pops, fits)
    print("ranks:", ranks)

    # 测试拥挤度排序算法
    dis = crowdingDistanceSort(pops, fits, ranks)
    print("dis:", dis)
