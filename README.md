# ML-ANC

## 基于机器学习的有源噪声控制论文写作代码

本程序主要研究基于机器学习的有源噪声控制算法的设计与测试，主要研究内容如下所示：

### 🌍 用到的的Github代码如下所示：

#### 🔊 音频处理相关仓库：

- 房间音频信号处理包：[LCAV/pyroomacoustics: Pyroomacoustics is a package for audio signal processing for indoor applications. It was developed as a fast prototyping platform for beamforming algorithms in indoor scenarios.](https://github.com/LCAV/pyroomacoustics) 📖 [API文档](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.datasets.html#)

- 音频目标质量和可解释性指标Python实现仓库**pysepm**：[schmiph2/pysepm: Python implementation of performance metrics in Loizou's Speech Enhancement book](https://github.com/schmiph2/pysepm)

- Meta-AF: 基于Meta-Learning的自适应滤波仓库：[adobe-research/MetaAF: Control adaptive filters with neural networks.](https://github.com/adobe-research/MetaAF)

- 滤波器设计包：FilterPy - Kalman filters and other optimal and non-optimal estimation filters in Python : [rlabbe/filterpy: Python Kalman filtering and optimal estimation library. Implements Kalman filter, particle filter, Extended Kalman filter, Unscented Kalman filter, g-h (alpha-beta), least squares, H Infinity, smoothers, and more. Has companion book 'Kalman and Bayesian Filters in Python'.](https://github.com/rlabbe/filterpy)
  - 🏠 [FilterPy — FilterPy 1.4.4 documentation](https://filterpy.readthedocs.io/en/latest/)

- **DeepFilterNet：**[Rikorose/DeepFilterNet: Noise supression using deep filtering](https://github.com/Rikorose/DeepFilterNet)

- 简化的Python音频特征提取包Spafe：[SuperKogito/spafe: spafe: Simplified Python Audio Features Extraction](https://github.com/SuperKogito/spafe) 📖 [API文档](https://superkogito.github.io/spafe/v0.2.0/api_documentation.html)

#### 🦄 多目标优化仓库

- ✨ **pymoo:** Multi-objective Optimization in Python : [anyoptimization/pymoo: NSGA2, NSGA3, R-NSGA3, MOEAD, Genetic Algorithms (GA), Differential Evolution (DE), CMAES, PSO](https://github.com/anyoptimization/pymoo)
  - 🏠 [pymoo 官网及手册](https://www.pymoo.org/)
  - 📖 [论文：Pymmo: Multi-Objective Optimization in Python](https://ieeexplore.ieee.org/document/9078759)

-  CMA-ES:[CMA-ES/pycma: Python implementation of CMA-ES](https://github.com/CMA-ES/pycma)

- ✨ Geatpy 2 ： The Genetic and Evolutionary Algorithm Toolbox for Python with high performance : [geatpy-dev/geatpy: Evolutionary algorithm toolbox and framework with high performance for Python](https://github.com/geatpy-dev/geatpy)
  - 🏠 [Geatpy](http://geatpy.com/)

- ✨ DEAP, a novel evolutionary cimputation frmework for rapid prototyping and testing of ideas : [DEAP/deap: Distributed Evolutionary Algorithms in Python](https://github.com/DEAP/deap)
  - 🏠 [DEAP documentation — DEAP 1.3.3 documentation](https://deap.readthedocs.io/en/master/)

- ✨ 多目标启发式算法 Github 仓库：[YuLi2022/MOEA-CODE-PYTHON: python实现多目标启发式算法](https://github.com/YuLi2022/MOEA-CODE-PYTHON)
  - 📖 1、MODA：多目标查分进化算法：[多目标差分进化在热连轧负荷分配中的应用 - 中国知网](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFD2010&filename=KZLY201007012&v=eACTLuuWgXCuSeHoCPfbYi6ACKx9earJFmPbFIEKL1eHZHWXctiVGXjkP5L0FVQO)
  - 🌐 2、NSGA2 ：[(56条消息) 多目标优化算法（一）NSGA-Ⅱ（NSGA2）_晓风wangchao的博客-CSDN博客_多目标优化算法](https://blog.csdn.net/qq_40434430/article/details/82876572)
  - 📖 3、MOPSO ：多目标粒子群算法：[MOPSO算法及其在水库优化调度中的应用 - 中国知网](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFD2007&filename=JSJC200718086&v=dGa1CTuXauWtahkDR3gOl6bdGGX8ycO6eRIycbCzkXYy2t91HEzutO66IGH%25mmd2BGf08)

- ✨ SPEA2 Github 仓库 (Strength Pareto Evolutionary Algorithm v2) : [Jagoslav/SPEA2: Strength Pareto Evolutionary Algorithm v2 implementation in python](https://github.com/Jagoslav/SPEA2)

- Pareto-hypernetworks : [AvivNavon/pareto-hypernetworks: Official implementation of Learning The Pareto Front With HyperNetworks [ICLR 2021]](https://github.com/AvivNavon/pareto-hypernetworks)


#### 📊 基于Pytorch的音频相关损失函数合计仓库：[csteinmetz1/auraloss: Collection of audio-focused loss functions in PyTorch](https://github.com/csteinmetz1/auraloss)
#### 控制系统仓库

- 基于 Python 的控制系统：[python-control/python-control: The Python Control Systems Library is a Python module that implements basic operations for analysis and design of feedback control systems.](https://github.com/python-control/python-control) 📖 [API文档](https://python-control.readthedocs.io/en/latest/intro.html)


### 📚 帮助文档：
- pyroomacoustics房间音频处理包帮助手册链接：[Contributing — Pyroomacoustics 0.6.0 documentation](https://pyroomacoustics.readthedocs.io/en/pypi-release/contributing.html)

💿 **音频数据集**：

- TIMIT音频数据集：[TIMIT Corpus — Pyroomacoustics 0.6.0 documentation](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.datasets.timit.html#the-timit-dataset)

- 回音消除研究数据集（AEC-Challenge）：[microsoft/AEC-Challenge: AEC Challenge](https://github.com/microsoft/AEC-Challenge) [相关论文：ICASSP 2021 Acoustic Echo Cancellation Challenge: Datasets, Testing Framework, and Results](https://arxiv.org/pdf/2009.04972.pdf)