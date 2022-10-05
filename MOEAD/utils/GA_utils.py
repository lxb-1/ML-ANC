from cmath import pi
import MOEAD_utils
import Draw_utils
import numpy as np

'''
遗传算法工具包
'''

def Creat_child(moead):
    # 创建一个个体
    child = moead.Test_fun.Bound[0] + (moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) * np.random.rand(
        moead.Test_fun.Dimention
    )
    return child

def Creat_Pop(moead):
    # 创建 moead.Pop_size 个种群
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

def mutate(moead, p1):
    # 突变个体的策略1
    var_num = moead.Test_fun.Dimention
    for i in range(int(var_num * 0.1)):
        d = moead.Test_fun.Bound[0] + (moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) * np.random.rand()
        d = d * np.random.randint(-1, 1)
        d = d / 10
        j = np.random.randint(0, var_num, size=1)[0]
        p1[j] = p1[j] + d
    return p1

def mutate2(moead, y1):
    # 突变个体的策略2
    dj = 0
    uj = np.random.rand()   # 生成一个 0-1 之间的一个随机实数
    if uj < 0.5:
        dj = (2 * uj) ** (1 / 6) - 1
    else:
        dj = 1 - 2 * (1 - uj) ** (1 / 6)
    y1 = y1 + dj
    y1[y1 > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
    y1[y1 < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
    return y1

def crossover(moead, pop1, pop2):
    # 交叉个体的策略1
    var_num = moead.Test_fun.Dimention
    r1 = int(var_num * np.random.rand())
    if np.random.rand() < 0.5:
        pop1[:r1], pop2[:r1] = pop2[:r1], pop1[:r1]
    else:
        pop1[r1:], pop2[r1:] = pop2[r1:], pop1[r1:]
    return pop1, pop2

def crossover2(moead, y1, y2):
    # 交叉个体的策略2
    var_num = moead.Test_fun.Dimention
    yj = 0
    uj = np.random.rand()
    if uj < 0.5:
        yj = (2 * uj) ** (1 / 3)
    else:
        yj = (1 / (2 * (1 - uj))) ** (1 / 3)
    y1 = 0.5 * (1 + yj) * y1 + (1 - yj) * y2
    y2 = 0.5 * (1 - yj) * y1 + (1 + yj) * y2
    y1[y1 > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
    y1[y1 < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
    y2[y2 > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
    y2[y2 < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
    return y1, y2

def EO(moead, wi, p1):
    m = p1.shape[0]
    tp_best = np.copy(p1)
    qbxf_tp = MOEAD_utils.cpt_tchebycheff(moead, wi, tp_best)
    Up = np.sqrt(moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) / 2
    h = 0
    for i in range(m):
        if h == 1:
            return tp_best
        temp_best = np.copy(p1)
        rd = np.random.normal(0, Up, 1)
        temp_best[i] = temp_best[i] + rd
        temp_best[temp_best > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
        temp_best[temp_best < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
        qbxf_te = MOEAD_utils.cpt_tchbycheff(moead, wi, temp_best)
        if qbxf_te < qbxf_tp:
            h = 1
            qbxf_tp = qbxf_te
            tp_best[:] = temp_best[:]
    return tp_best

def cross_mutation(moead, p1, p2):
    y1 = np.copy(p1)
    y2 = np.copy(p2)
    c_rate = 1      # 交叉率？
    m_rate = 0.5    # 突变率？
    if np.random.rand() < c_rate:
        y1, y2 = crossover2(moead, y1, y2)
    if np.random.rand() < m_rate:
        y1 = mutate2(moead, y1)
        y2 = mutate2(moead, y2)
    return y1, y2

def generate_next(moead, gen, wi, p0, p1, p2):
    # 进化下一代个体：基于自身 Xi+邻域内随机选择的两个 Xk, Xl 来考虑进化下一代
    qbxf_p0 = MOEAD_utils.cpt_tchebycheff(moead, wi, p0)
    qbxf_p1 = MOEAD_utils.cpt_tchebycheff(moead, wi, p1)
    qbxf_p2 = MOEAD_utils.cpt_tchebycheff(moead, wi, p2)

    qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2])
    best = np.argmin(qbxf)
    # 选中切比雪夫距离最小（最好的）的个体
    Y1 = [p0, p1, p2][best]
    # 深拷贝成独立一份
    n_p0, n_p1, n_p2 = np.copy(p0), np.copy(p1), np.copy(p2)

    if gen % 10 == 0:
        # 每隔 10 代，有小概率进行 EO 优化（效果号，但是复杂度高）
        if np.random.rand() < 0.1:
            np_p0 = EO(moead, wi, np_p0)
    # 交叉
    n_p0, n_p1 = cross_mutation(moead, n_p0, n_p1)
    n_p1, n_p2 = cross_mutation(moead, n_p1, n_p2)
    # 交叉后的切比雪夫距离
    qbxf_np0 = MOEAD_utils.cpt_tchbycheff(moead, wi, n_p0)
    qbxf_np1 = MOEAD_utils.cpt_tchbycheff(moead, wi, n_p1)
    qbxf_np2 = MOEAD_utils.cpt_tchbycheff(moead, wi, n_p2)

    qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2, qbxf_np0, qbxf_np1, qbxf_np2])
    best = np.argmin(qbxf)
    # 选中切比雪夫距离最小（最好的）个体
    Y2 = [p0, p1, p2, n_p0, n_p1, n_p2][best]

    # 随机选中目标中的某一个目标进行判断
    fm = np.random.randint(0, moead.Test_fun.Func_num)
    # 如果是极小化目标求解，以 0.5 的概率进行更详细的判断。
    # 注意：返回最优解解策略不能太死板，否则容易陷入局部最优
    if moead.problem_type == 0 and np.random.rand() < 0.5:
        FY1 = moead.Test_fun.Func(Y1)
        FY2 = moead.Test_fun.Func(Y2)
        # 如果随机选择的 Y2 目标更好，则返回 Y2 的目标
        if FY2[fm] < FY1[fm]:
            return Y2
        else:
            return Y1
    return Y2

def evolution(moead):
    # 进化，开始进化 moead.max_gen 轮
    for gen in range(moead.max_gen):
        # 用于图像展示时候，告诉它现在在第几轮了
        moead.gen = gen
        # 取出群体数组 moead.Pop 中的每一个个体，进行进化
        # pi 表示个体序号，其中， p——personal，i——index
        Bi = moead.W_Bi_T[pi]
        # 随机选取一个 T 内的数，作为 pi 的邻域
        # （邻域你可以想象成：物种，你总不能人狗杂交吧？所以个体 pi 只能与他的 T个 前后的邻居权重，管的个体杂交进化）
        # ？比如：T=2，权重(0.1,0.9)约束的个体的邻居是：权重(0,1)、(0.2,0.8)约束的个体。永远固定不变
        k = np.random.randint(moead.T_size)
        l = np.random.randint(moead.T_size)
        # 随机从邻域中选择两个个体，产生新解
        ik = Bi[k]
        il = Bi[l]
        Xi = moead.Pop[pi]
        Xk = moead.Pop[ik]
        Xl = moead.Pop[il]
        # 进化下一代个体：基于自身 Xi + 邻域中随机选择的 2 个 Xk、Xl，并结合 gen 进行进化
        Y = generate_next(moead, gen, pi, Xi, Xk, Xl)
        # 计算进化前 Xi 的切比雪夫距离
        cbxf_i = MOEAD_utils.cpt_tchebycheff(moead, pi, Xi)
        # 计算进化后 Xi 的切比雪夫距离
        cbxf_y = MOEAD_utils.cpt_tchebycheff(moead, pi, Y)
        # 设置 Stop Criteria
        d = 0.001
        # 开始比较是否进化出了更好的下一代
        if cbxf_y < cbxf_i:
            # 用于绘图：当前进化种群中，哪个正在被进化；draw_w = true 的时候才可见
            moead.now_y = pi
            # 计算下一代的函数值
            F_Y = moead.Test_fun.Func(Y)[:]
            # 更新函数值到 moead.EP_X_FV 中。拥有了更好的切比雪夫下一代，自然要更新多目标中的目标函数值
            MOEAD_utils.update_EP_By_ID(moead, pi, F_Y)