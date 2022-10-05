import os
from math import sqrt

import numpy as np
from Mean_Vector_utils import Mean_vector

"""
MOEAD 工具包
"""

def Load_W(moead):
    file = moead.name + '.csv'
    path = moead.csv_file_path + '/' + file

    # 如果不存在则创建，如果存在则直接读取到 W 中
    if os.path.exists(path) == False:
        print('not exists')
        # 如果不存在权重矩阵，则构建平均向量 mv
        # mv = Mean_vector(moead.h, moead.Test_fun.Func_num, path)
        mv = Mean_vector(
            moead.h,                    # 目标方向个数
            moead.Test_fun.Func_num,    # 空间维度
            path                        # 均值项链不过的存储位置
        )
        # 调用 mv 的类方法 generate() 方法，计算向量平均，并以 ndarray 格式存储到 path 指定的位置
        mv.generate()
        print("created")
    W = np.loadtxt(fname=path)
    moead.Pop_size = W.shape[0]
    moead.W = W
    return W

def cpt_Z(moead):
    # 函数使用说明：
    # 初始化 Z 集，最小问题 0, 0, ...
    # 实验结论 : 如果已经知道优化目标，比如极小化问题，理想极小值为 [0, 0]
    # 那你就一开始的时候就写死moead.Z=[0,0]吧，就不用这个函数进行设置了。
    # ！！！ 这里极小化全部初始化为[0,0,...0]，极大化：[10,10,....10]
    Z = []
    for fi in range(moead.Test_fun.Func_num):
        z_i = -1
        if moead.problem_type == 0:
            z_i = 0
        if moead.problem_type == 1:
            z_i = 10
        Z.append(z_i)
    moead.Z = Z
    return Z

def cpt_Z2(moead):
    # 初始化 Z 集，最小问题 0, 0, ...
    # FV 表示 F-值吗？
    Z = moead.Pop_FV[0][:]
    dz = np.random.rand()
    for fi in range(moead.Test_fun.Func_num):
        for Fpi in moead.Pop_FV:
            if moead.problem_type == 0:
                if Fpi[fi] < Z[fi]:
                    Z[fi] = Fpi[fi] - dz
            if moead.problem_type == 1:
                if Fpi[fi] > Z[fi]:
                    Z[fi] = Fpi[fi] + dz
    moead.Z = Z
    return Z

def init_EP(moead):
    # 计算初始化前沿
    for pi in range(moead.Pop_size):
        np = 0
        F_V_P = moead.Pop_FV[pi]
        for ppi in range(moead.Pop_size):
            F_V_PP = moead.Pop_FV[ppi]
            if pi != ppi:
                # 判断 F_V_PP 是否支配 F_V_P
                if is_dominate(moead, F_V_PP, F_V_P):
                    np += 1
        if np == 0:
            moead.EP_X_ID.append(pi)
            moead.EP_X_FV.append(F_V_P[:])

def cpt_W_Bi_T(moead):
    # 计算权重的 T 个邻居
    if moead.T_size < 1:
        return -1
    for bi in range(moead.W.shape[0]):
        Bi = moead.W[bi]
        DIS = np.sum((moead.w - Bi) ** 2, axis=1)
        # 对 DIS 进行排序，并返回排序后的索引到 B_T 变量中
        B_T = np.argsort(DIS)
        # 第 0 个是自己（距离永远最小）
        B_T = B_T[1:moead.T_size + 1]
        moead.W_Bi_T.append(B_T)

def is_dominate(moead, F_X, F_Y):
    # 判断 F_X 是否支配 F_Y
    if type(F_Y) != list:
        F_X = moead.Test_fun.Func(F_X)
        F_Y = moead.Test_fun.Func(F_Y)
    i = 0
    if moead.problem_type == 0:     # 最小化问题
        for xv, yv in zip(F_X, F_Y):
            if xv < yv:
                i = i + 1
            if xv > yv:
                return False
    if moead.problem_type == 1:     # 最大化问题
        for xv, yv in zip(F_X, F_Y):
            if xv > yv:
                i = i + 1
            if xv < yv:
                return False
    if i != 0:
        return True
    return False

def cpt_to_Z_dist(moead, X):
    # 计算 X 点到参考点的距离
    F_X = moead.Test_fun.Func(X)
    d = 0
    for i, fm in enumerate(F_X):
        d = d + (fm - moead.Z[i]) ** 2
    d = sqrt(d)
    return d

def Tchebycheff_dist(w, f, z):
    # 计算切比雪夫距离
    return w * abs(f - z)

def cpt_tchebycheff(moead, idx, X):
    # idx : X 在种群中的位置
    # 计算 X 的切比雪夫距离，即与理想点 Z 的距离
    max = moead.Z[0]
    ri = moead.W[idx]
    F_X = moead.Test_fun.Func(X)
    for i in range(moead.Test_fun.Func_num):
        fi = Tchebycheff_dist(ri[i], F_X[i], moead.Z[i])
        if fi > max:
            max = fi
    return max

def update_BTX(moead, P_B, Y):
    # 根据 Y 更新 P_B 集内邻居
    for j in P_B:
        Xj = moead.Pop[j]
        d_x = cpt_tchebycheff(moead, j, Xj)
        d_y = cpt_tchebycheff(moead, j, Y)
        if d_y <= d_x:
            # d_y 的切比雪夫距离更小
            moead.Pop[j] = Y[:]
            F_Y = moead.Test_fun.Func(Y)
            moead.Pop_FV[j] = F_Y
            update_EP_By_ID(moead, j, F_Y)

def update_EP_By_ID(moead, id, F_Y):
    # 如果 id 存在，则更新其对应函数集合的值
    if id in moead.EP_X_ID:
        # 拿到所在的位置
        position_pi = moead.EP_X_ID.index(id)
        # 更新函数值
        moead.EP_X_FV[position_pi][:] = F_Y[:]

def update_Z(moead, Y):
    # 根据 Y 更新 Z 坐标。
    dz = np.random.rand()
    F_y = moead.Test_fun.Func(Y)
    for j in range(moead.Test_fun.Func_num):
        if moead.problem_type == 0:     # 极小化
            if moead.Z[j] > F_y[j]:
                moead.Z[j] = F_y[j] - dz
        if moead.problem_type == 1:     # 极大化
            if moead.Z[j] < F_y[j]:
                moead.Z[j] = F_y[j] + dz

def update_EP_By_Y(moead, id_Y):
    # 根据 Y 更新前沿
    # 根据 Y 更新 EP
    i = 0

    F_Y = moead.Pop_FV[id_Y]    # 拿到 id_Y 的函数值
    Delet_set = []              # 需要被删除的集合
    Len = len(moead.EP_X_FV)    # 支配前沿集合的数量
    for pi in range(Len):
        # F_Y 是否支配 pi 号个体，如果支配，则提出 pi 号个体
        if is_dominate(moead, F_Y, moead.EP_X_FV[pi]):
            # 列入被删除的集合
            Delet_set.append(pi)
            break
        if i != 0:
            break
        if is_dominate(moead, moead.EP_X_FV[pi], F_Y):
            # 它有被别人支配，然后记下来能支配它的个数
            i += 1
    # 新的支配前沿的 ID 集合，种群个体 ID
    new_EP_X_ID = []
    # 新的支配前沿集合的函数值
    new_EP_X_FV = []
    for save_id in range(Len):
        if save_id not in Delet_set:
            # 不在 Delet_set 中，则保存
            new_EP_X_ID.append(moead.EP_X_ID[save_id])
            new_EP_X_FV.append(moead.EP_X_FV[save_id])
    # 更新上面计算号的新的支配前沿
    moead.EP_X_ID = new_EP_X_ID
    moead.EP_X_FV = new_EP_X_FV
    # 如果 i == 0，一位置没有人支配 id_Y
    # 没人支配 id_Y，就将其加入支配前沿
    if i == 0:
        # 不在里面直接添加新成员
        if id_Y not in moead.EP_X_ID:
            moead.EP_X_ID.append(id_Y)
            moead.EP_X_FV.append(F_Y)
        else:
            # 本来就在里面，则更新即可
            idy = moead.EP_X_ID.index(id_Y)
            moead.EP_X_FV[idy] = F_Y[:]
    return moead.EP_X_ID, moead.EP_X_FV