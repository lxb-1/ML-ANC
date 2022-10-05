import time

# 导入测试函数
import problem.ZDT1 as ZDT1

# 工具包
from utils.Draw_utils import *
import utils.GA_utils as GA_utils
import utils.DE_utils as DE_utils
from utils.Mean_Vector_utils import *
from utils.MOEAD_utils import *

class MOEAD:
    problem_type  = 0           # 0 表示最小化目标求解， 1 表示最大化目标求解
    Test_fun      = ZDT1        # 测试函数
    name          = 'ZDT1'      # 动态展示时的 `title` 名称
    GA_DE_Utils   = GA_utils    # 可用的进化算法：(1) 遗传算法 GA_utils (2) 查分进化算法 DE_utils
    Pop_size      = -1          # 种群大小，取决于 `Vector` 文件夹下的 `xx.csv`
    max_gen       = 50          # 最大迭代数
    T_size        = 5           # 邻域数 （即只对邻域内的相互更新、交叉）
    EP_X_ID       = []          # 支配前沿 (Pareto Front??) ID
    EP_X_FV       = []          # 支配前沿 (Pareto Front??) 函数值
    Pop           = []          # 种群
    Pop_FV        = []          # 种群计算出来的函数
    W             = []          # 权重
    W_Bi_T        = []          # 权重的 T 个邻居，比如，在 T = 2 时, (0.1, 0.9) 的邻域为： (0, 1)、(0.2, 0.8)。永远固定不变
    Z             = []          # 理想点（对于最小化问题，理想点趋于 0），对于具有两个目标的极小化问题，Z = [0, 0]
    csv_file_path = 'Vector'    # 权重向量存储目录
    gen           = 0           # 当前迭代的代数
    need_dynamic  = True        # 是否动态展示
    draw_w        = True        # 是否绘制权重图
    now_y         = []          # 用于绘图：当前进化种群中，哪个这个在进化

    def __init__(self):
        self.Init_data()

    def Init_data(self):
        Load_W(self)            # 加载权重
        cpt_W_Bi_T(self)        # 计算每个权重 Wi 的 T 个邻域
        self.GA_DE_Utils.Creat_Pop(self)    # 创建种群
        cpt_Z(self)             # 初始化 Z 集，最小化问题为 0, 0

    def show(self):
        if self.draw_w:
            draw_W(self)
        draw_MOEAD_Pareto(self, moead.name + "num:" + str(self.max_gen) + "")
        show()

    def run(self):
        t = time.time()
        # EP_X_ID：支配前沿个体解，的ID。在上面数组：Pop，中的序号
        # envolution开始进化
        EP_X_ID = self.GA_DE_Utils.evolution(self)
        print('你拿以下序号到上面数组：Pop中找到对应个体，就是多目标优化的函数的解集啦!')
        print("支配前沿个体解，的ID（在上面数组：Pop，中的序号）：", EP_X_ID)
        dt = time.time() - t
        self.show()

if __name__ == '__main__':
    moead = MOEAD()
    moead.run()