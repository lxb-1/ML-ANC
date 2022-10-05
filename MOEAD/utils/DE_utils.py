import MOEAD_utils
import Draw_utils
import numpy as np

'''
差分进化算法工具包
'''

# 交叉率
Cross_Rate = 0.5

def Creat_child(moead):
    # 创建一个个体：即一个向量，长度为 Dimention，范围在 moead.Test_fun.Bound 中设定
    child = moead.Test_fun.Bound[0] + (moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) * np.random.rand(
        moead.Test_fun.Dimention
    )
    return child

def Creat_Pop(moead):
    Pop = []
    Pop_FV = []
    if moead.Pop_size < 1:
        print('error in creat_Pop')
        return -1
    while len(Pop) != moead.Pop_size:
        X = Creat_child(moead)
        Pop.append(X)
        Pop_FV.append(moead.Test_fun.Func(X))
    moead.Pop, moead.Pop_FV = Pop, Pop_FV
    return Pop, Pop_FV

def mutate(moead, best, p1, p2):
    f = 0.5 + 1.5 * np.random.rand()    # 缩放因子
    d = f * (p1 - p2)
    temp_p = best + d
    temp_p[temp_p > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[0] + (
            moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) * np.random.rand()
    temp_p[temp_p < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0] + (
            moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) * np.random.rand()
    return temp_p

def crossover(moead, p1, vi):
    var_num = moead.Test_fun.Dimention
    ui = np.zeros(var_num)
    k = np.random.random_integers(0, var_num - 1)
    for j in range(0, var_num):
        if np.random.random() < Cross_Rate or j == k:
            ui[j] = vi[j]
        else:
            ui[j] = p1[j]
    return ui

def generate_next(moead, wi, p0, p1, p2):
    qbxf_p0 = MOEAD_utils.cpt_tchebycheff(moead, wi, p0)
    qbxf_p1 = MOEAD_utils.cpt_tchebycheff(moead, wi, p1)
    qbxf_p2 = MOEAD_utils.cpt_tchebycheff(moead, wi, p2)
    arr = [p0, p1, p2]
    qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2])
    index = np.argsort(qbxf)
    best = arr[index[0]]
    bw = arr[index[2]]
    bm = arr[index[1]]

    vi = mutate(moead, best, bm, bw)
    ui = crossover(moead, p0, vi)
    return ui

def evolution(moead):
    for gen in range(moead.max_gen):
        moead.gen = gen
        for pi, p in enumerate(moead.Pop):
            # 第 pi 个个体的邻域集合
            Bi = moead.W_Bi_T[pi]
            k = np.random.randint(moead.T_size)
            l = np.random.randint(moead.T_size)
            # 随机从邻域内选择 2 个个体，产生新解
            ik = Bi[k]      # 随机获得的 Bi 中的编号
            il = Bi[l]      # 随机获得的 Bi 中的编号
            # 由编号获得种群
            Xi = moead.Pop[pi]
            Xk = moead.Pop[ik]
            Xl = moead.Pop[il]

            Y = generate_next(moead, pi, Xi, Xk, Xl)
            # 下面根据切比雪夫方法，分别计算原始种群与进化种群的优越性
            cbxf_i = MOEAD_utils.cpt_tchebycheff(moead, pi, Xi)     # 原始种群 Xi 的切比雪夫距离，即与理想点 Z 的距离
            cbxf_y = MOEAD_utils.cpt_tchebycheff(moead, pi, Y)      # 进化种群 Xi 的切比雪夫距离，即与理想点 Z 的距离

            # 设置终止准则
            d = 0.001
            if cbxf_y < cbxf_i:     # 如果进化种群比原始种群更优越
                moead.now_y = pi
                moead.Pop[pi] = np.copy(Y)          # 将更优越的进化种群存储到原始数据对应位置 moead.Pop[i]
                F_Y = moead.Test_fun.Func(Y)[:]     # ？
                # 更新函数值到 moead.EP_X_FV 中。拥有了更好的切比雪夫下一代，自然要更新多目标中的目标函数值
                MOEAD_utils.update_EP_By_ID(moead, pi, F_Y)
                # 进化出更好的切比雪夫下一代了，有可能有更好的理想点了，尝试更新新的理想点
                MOEAD_utils.update_Z(moead, Y)
                if abs(cbxf_y - cbxf_i) > d:
                    MOEAD_utils.update_EP_By_Y(moead, pi)
            # 根据 Y 种群更新 P_B 集内邻居
            MOEAD_utils.update_BTX(moead, Bi, Y)

        if moead.need_dynamic:
            Draw_utils.plt.cla()
            if moead.draw_w:
                Draw_utils.draw_W(moead)
            Draw_utils.draw_MOEAD_Pareto(moead, moead.name + "第：" + str(gen) + "")
            Draw_utils.plt.pause(0.001)
        print('gen %s,EP size :%s,Z:%s' % (gen, len(moead.EP_X_ID), moead.Z))
    return moead.EP_X_ID