import numpy as np

'''
求解均值向量工具包
'''

class Mean_vector:
    """
    类描述 : 均值向量类

    参数:
        m : 空间维度
        H : 目标方向个数
        path : 均值向量的存储位置
    """
    def __init__(self, H=5, m=3, path='out.csv'):
        self.H        = H
        self.m        = m
        self.path     = path
        self.stepsize = 1 / H

    def perm(self, sequence):
        """序列全排列，且无重复

        Args:
            sequence (_type_): _description_

        返回值：
            r (列表) : 排序后的序列，以列表形式排列
        """
        l = sequence
        if (len(l) <= 1):
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i-1] == sequence[i]:
                continue
            else:
                s = l[:i] + l[i+1:]
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i+1] + x)
        return r

    def get_mean_vectors(self):
        """计算向量平均

        Returns:
            ws (列表) : 加权系数矩阵 (WS, Weight Sum)
        """
        H = self.H      # 目标方向个数
        m = self.m      # 空间维度
        sequence = []
        # 经过下面两个 for 循环后，sequence 变成了 [0, 0, 0, 0, 0, 1, 1]
        for ii in range(H):
            sequence.append(0)
        for jj in range(m-1):
            sequence.append(1)
        # 加权系数矩阵
        ws = []

        # 使用序列全排列类方法 perm() 方法获取 sequence 的全排列组合 pe_seq
        pe_seq = self.perm(sequence)
        # 遍历 sequence 的全排列 pe_seq 中的每个组合
        for sq in pe_seq:
            s = -1
            weight = []

            # 对每个组合 sq 设置权重 weight
            for i in range(len(sq)):
                if sq[i] == 1:
                    w = i - s
                    w = (w-1) / H
                    s = i
                    weight.append(w)
            nw = H + m -1 - s
            nw = (nw-1) / H
            weight.append(nw)
            if weight not in ws:
                ws.append(weight)
        return ws

    def save_mv_to_file(self, mv):
        """存储平均向量到指定文件

        Args:
            mv (_type_): 需要存储的平均向量
        """
        f = np.array(mv, dtype=np.float64)
        np.savetxt(fname=self.path, X=f)

    def generate(self):
        m_v = self.get_mean_vectors()
        self.save_mv_to_file(m_v)


