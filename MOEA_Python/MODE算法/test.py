# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2020/11/04
@Author  :   Yu Li
@describe:   使用fun算例测试多目标MODE
'''

from cProfile import label
import matplotlib.pyplot as plt
from MODE import *
from function import *
from fitness import *

def main():
    nIter = 100
    nChr = 3
    nPop = 50
    F = 0.2
    Cr = 0.9
    func = function
    lb = -2
    rb = 2

    # 调用 MODE 函数计算 Pareto 种群与 Pareto 前沿
    paretoPops, paretoFits = MODE(nIter, nChr, nPop, F, Cr, func, lb, rb)
    # 查看 Pareto 前沿的形状
    print(f"paretoFront: {paretoFits.shape}")

    # 理论最优解集合
    x = np.linspace(-1 / np.sqrt(3), 1 / np.sqrt(3), 116).reshape(116, 1)
    X = np.tile(x, 3)   # 理论最优 Pareto 解集
    thFits = fitness(X, function)

    # 如果显示中文可以添加如下代码
    # plt.rcParams['font.sans-serif'] = 'KaiTi'   # 正楷

    # 绘图
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)
    ax.plot(thFits[:,0], thFits[:,1], color='green', \
        label="Theory Pareto Fron")     # 绘制理论 Pretor 前沿
    ax.scatter(paretoFits[:, 0], paretoFits[:, 1], color='red', \
        label="Practical solution")     # 绘制 MODE 算法计算的 Pareto 前沿
    ax.legend()     # 绘制图例
    plt.show()      # 显示绘制结果
    fig.savefig('test.jpg', dpi=400)    # 存储绘制结果，我这里目前不好使？？

if __name__ == "__main__":
    main()