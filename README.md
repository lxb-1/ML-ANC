# ML-ANC

## 基于机器学习的有源噪声控制论文写作代码

本程序主要研究基于机器学习的有源噪声控制算法的设计与测试，主要研究内容如下所示：

### 🌍 用到的的Github代码如下所示：

#### 🔊 音频处理相关仓库：

- 房间音频信号处理包：[LCAV/pyroomacoustics: Pyroomacoustics is a package for audio signal processing for indoor applications. It was developed as a fast prototyping platform for beamforming algorithms in indoor scenarios.](https://github.com/LCAV/pyroomacoustics) 📖 [API文档](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.datasets.html#)

- 音频目标质量和可解释性指标Python实现仓库**pysepm**：[schmiph2/pysepm: Python implementation of performance metrics in Loizou's Speech Enhancement book](https://github.com/schmiph2/pysepm)

- **DeepFilterNet：**[Rikorose/DeepFilterNet: Noise supression using deep filtering](https://github.com/Rikorose/DeepFilterNet)

- 简化的Python音频特征提取包Spafe：[SuperKogito/spafe: spafe: Simplified Python Audio Features Extraction](https://github.com/SuperKogito/spafe) 📖 [API文档](https://superkogito.github.io/spafe/v0.2.0/api_documentation.html)

- Play and Record Sound with Python：[spatialaudio/python-sounddevice: Play and Record Sound with Python](https://github.com/spatialaudio/python-sounddevice) 📖 [API文档](https://python-sounddevice.readthedocs.io/en/0.4.5/index.html)

- [pyannote/pyannote-audio: Neural building blocks for speaker diarization: speech activity detection, speaker change detection, overlapped speech detection, speaker embedding](https://github.com/pyannote/pyannote-audio) [手册](https://pyannote.github.io/pyannote-core/structure.html)

- Meta-AF: Meta-Learning for Adaptive Filters ([2022 arXiv](https://arxiv.org/pdf/2204.11942.pdf)) ：[adobe-research/MetaAF: Control adaptive filters with neural networks.](https://github.com/adobe-research/MetaAF#demos)

- `LEAF`: A Learnable Audio Frontend : [google-research/leaf-audio: LEAF is a learnable alternative to audio features such as mel-filterbanks, that can be initialized as an approximation of mel-filterbanks, and then be trained for the task at hand, while using a very small number of parameters.](https://github.com/google-research/leaf-audio)

- [smitkiri/urban-sound-classification: Classification of audio signals using PyTorch](https://github.com/smitkiri/urban-sound-classification)
- 
- python_speech_features仓库：[jameslyons/python_speech_features: This library provides common speech features for ASR including MFCCs and filterbank energies.](https://github.com/jameslyons/python_speech_features)
- [sambittarai/Audio-Signal-Processing-using-Deep-Learning: This repository includes an entire workflow for Audio Classification using Deep Learning.](https://github.com/sambittarai/Audio-Signal-Processing-using-Deep-Learning)

- ✨ WavEncoder:[shangeth/wavencoder: WavEncoder is a Python library for encoding audio signals, transforms for audio augmentation, and training audio classification models with PyTorch backend.](https://github.com/shangeth/wavencoder)。它设计了很多特有的块和层，可以参考文献中关于WavEncoder的相关文献。

- ✨ 音频网络网络架构SincNet:[mravanelli/SincNet: SincNet is a neural architecture for efficiently processing raw audio samples.](https://github.com/mravanelli/SincNet)

- **Speexdsp-python**：[xiongyihui/speexdsp-python: Speex Echo Canceller Python Library](https://github.com/xiongyihui/speexdsp-python)

- 语言识别到文本实现以及在线深度学习实现仓库（36.4k star）：[CorentinJ/Real-Time-Voice-Cloning: Clone a voice in 5 seconds to generate arbitrary speech in real-time](https://github.com/CorentinJ/Real-Time-Voice-Cloning) 作者硕士论文链接：[Master thesis : Automatic Multispeaker Voice Cloning - s123578Jemine2019.pdf](https://matheo.uliege.be/bitstream/2268.2/6801/5/s123578Jemine2019.pdf)

- 音乐音频python分析包：[librosa/librosa: Python library for audio and music analysis](https://github.com/librosa/librosa)

- PyTorch-Kaldi语音识别工具包：[mravanelli/pytorch-kaldi: pytorch-kaldi is a project for developing state-of-the-art DNN/RNN hybrid speech recognition systems. The DNN part is managed by pytorch, while feature extraction, label computation, and decoding are performed with the kaldi toolkit.](https://github.com/mravanelli/pytorch-kaldi)

- （FaceBook）前沿序列建模工具包（Fairseq）：[facebookresearch/fairseq: Facebook AI Research Sequence-to-Sequence Toolkit written in Python.](https://github.com/facebookresearch/fairseq)

- Speechbrain：[speechbrain/speechbrain: A PyTorch-based Speech Toolkit](https://github.com/speechbrain/speechbrain)

#### 🦄 多目标优化仓库

- 🎞️ 多目标优化相关的视频：
  - ✨ [多目标优化_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV17S4y1M7oU/?spm_id_from=333.337.search-card.all.click&vd_source=5a45e7b7d8a4889aab6645f3fbfd5fee)

- ✨ **pymoo:** Multi-objective Optimization in Python : [anyoptimization/pymoo: NSGA2, NSGA3, R-NSGA3, MOEAD, Genetic Algorithms (GA), Differential Evolution (DE), CMAES, PSO](https://github.com/anyoptimization/pymoo)
  - 🏠 [pymoo 官网及手册](https://www.pymoo.org/)
  - 📖 [论文：Pymmo: Multi-Objective Optimization in Python](https://ieeexplore.ieee.org/document/9078759)

- ✨ Geatpy 2 ： The Genetic and Evolutionary Algorithm Toolbox for Python with high performance : [geatpy-dev/geatpy: Evolutionary algorithm toolbox and framework with high performance for Python](https://github.com/geatpy-dev/geatpy)
  - 🏠 [Geatpy](http://geatpy.com/)

- ✨ DEAP, a novel evolutionary cimputation frmework for rapid prototyping and testing of ideas : [DEAP/deap: Distributed Evolutionary Algorithms in Python](https://github.com/DEAP/deap)
  - 🏠 [DEAP documentation — DEAP 1.3.3 documentation](https://deap.readthedocs.io/en/master/)

- ✨ 多目标启发式算法 Github 仓库：[YuLi2022/MOEA-CODE-PYTHON: python实现多目标启发式算法](https://github.com/YuLi2022/MOEA-CODE-PYTHON)
  - 📖 1、MODA：多目标查分进化算法：[多目标差分进化在热连轧负荷分配中的应用 - 中国知网](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFD2010&filename=KZLY201007012&v=eACTLuuWgXCuSeHoCPfbYi6ACKx9earJFmPbFIEKL1eHZHWXctiVGXjkP5L0FVQO)
  - 🌐 2、NSGA2 ：[(56条消息) 多目标优化算法（一）NSGA-Ⅱ（NSGA2）_晓风wangchao的博客-CSDN博客_多目标优化算法](https://blog.csdn.net/qq_40434430/article/details/82876572)
  - 📖 3、MOPSO ：多目标粒子群算法：[MOPSO算法及其在水库优化调度中的应用 - 中国知网](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFD2007&filename=JSJC200718086&v=dGa1CTuXauWtahkDR3gOl6bdGGX8ycO6eRIycbCzkXYy2t91HEzutO66IGH%25mmd2BGf08)

- ✨ SPEA2 Github 仓库 (Strength Pareto Evolutionary Algorithm v2) : [Jagoslav/SPEA2: Strength Pareto Evolutionary Algorithm v2 implementation in python](https://github.com/Jagoslav/SPEA2)

- 寻找一个大量点集的 `pareto front` 的快速实现方法仓库 ：[KoenGoe/FastPareto: Fast implementations for finding pareto front in set of points](https://github.com/KoenGoe/FastPareto)

- Pareto-hypernetworks : [AvivNavon/pareto-hypernetworks: Official implementation of Learning The Pareto Front With HyperNetworks [ICLR 2021]](https://github.com/AvivNavon/pareto-hypernetworks)

- [jMetal/jMetalPy: A framework for single/multi-objective optimization with metaheuristics](https://github.com/jMetal/jMetalPy#installation) 📚 [jMetal/jMetaalPY帮助文档](https://jmetal.github.io/jMetalPy/tutorials/observer.html)

#### 🎛️ 并行计算相关仓库：
- Dask : A flexible parallel computing library for analytics : [dask/dask: Parallel computing with task scheduling](https://github.com/dask/dask)
  - 📚 [Dask 文档](https://docs.dask.org/en/latest/install.html)
  - 🏠 [Dask 官网](https://www.dask.org/)

#### 📊 基于Pytorch的音频相关损失函数合计仓库：[csteinmetz1/auraloss: Collection of audio-focused loss functions in PyTorch](https://github.com/csteinmetz1/auraloss)

相关论文链接如下表所示：

<table>
    <tr>
        <th>Loss function</th>
        <th>Interface</th>
        <th>Reference</th>
    </tr>
    <tr>
        <td colspan="3" align="center"><b>Time domain</b></td>
    </tr>
    <tr>
        <td>Error-to-signal ratio (ESR)</td>
        <td><code>auraloss.time.ESRLoss()</code></td>
        <td><a href=https://arxiv.org/abs/1911.08922>Wright & Välimäki, 2019</a></td>
    </tr>
    <tr>
        <td>DC error (DC)</td>
        <td><code>auraloss.time.DCLoss()</code></td>
        <td><a href=https://arxiv.org/abs/1911.08922>Wright & Välimäki, 2019</a></td>
    </tr>
    <tr>
        <td>Log hyperbolic cosine (Log-cosh)</td>
        <td><code>auraloss.time.LogCoshLoss()</code></td>
        <td><a href=https://openreview.net/forum?id=rkglvsC9Ym>Chen et al., 2019</a></td>
    </tr>
    <tr>
        <td>Signal-to-noise ratio (SNR)</td>
        <td><code>auraloss.time.SNRLoss()</code></td>
        <td></td>
    </tr>
    <tr>
        <td>Scale-invariant signal-to-distortion <br>  ratio (SI-SDR)</td>
        <td><code>auraloss.time.SISDRLoss()</code></td>
        <td><a href=https://arxiv.org/abs/1811.02508>Le Roux et al., 2018</a></td>
    </tr>
    <tr>
        <td>Scale-dependent signal-to-distortion <br>  ratio (SD-SDR)</td>
        <td><code>auraloss.time.SDSDRLoss()</code></td>
        <td><a href=https://arxiv.org/abs/1811.02508>Le Roux et al., 2018</a></td>
    </tr>
    <tr>
        <td colspan="3" align="center"><b>Frequency domain</b></td>
    </tr>
    <tr>
        <td>Aggregate STFT</td>
        <td><code>auraloss.freq.STFTLoss()</code></td>
        <td><a href=https://arxiv.org/abs/1808.06719>Arik et al., 2018</a></td>
    </tr>
    <tr>
        <td>Aggregate Mel-scaled STFT</td>
        <td><code>auraloss.freq.MelSTFTLoss(sample_rate)</code></td>
        <td></td>
    </tr>
    <tr>
        <td>Multi-resolution STFT</td>
        <td><code>auraloss.freq.MultiResolutionSTFTLoss()</code></td>
        <td><a href=https://arxiv.org/abs/1910.11480>Yamamoto et al., 2019*</a></td>
    </tr>
    <tr>
        <td>Random-resolution STFT</td>
        <td><code>auraloss.freq.RandomResolutionSTFTLoss()</code></td>
        <td><a href=https://www.christiansteinmetz.com/s/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf>Steinmetz & Reiss, 2020</a></td>
    </tr>
    <tr>
        <td>Sum and difference STFT loss</td>
        <td><code>auraloss.freq.SumAndDifferenceSTFTLoss()</code></td>
        <td><a href=https://arxiv.org/abs/2010.10291>Steinmetz et al., 2020</a></td>
    </tr>
    <tr>
        <td colspan="3" align="center"><b>Perceptual transforms</b></td>
    </tr>
    <tr>
        <td>Sum and difference signal transform</td>
        <td><code>auraloss.perceptual.SumAndDifference()</code></td>
        <td><a href=#></a></td>
    </tr>
    <tr>
        <td>FIR pre-emphasis filters</td>
        <td><code>auraloss.perceptual.FIRFilter()</code></td>
        <td><a href=https://arxiv.org/abs/1911.08922>Wright & Välimäki, 2019</a></td>
    </tr>
</table>

\* [Wang et al., 2019](https://arxiv.org/abs/1904.12088) also propose a multi-resolution spectral loss (that [Engel et al., 2020](https://arxiv.org/abs/2001.04643) follow), 
but they do not include both the log magnitude (L1 distance) and spectral convergence terms, introduced in [Arik et al., 2018](https://arxiv.org/abs/1808.0671), and then extended for the multi-resolution case in [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480).


#### 控制系统仓库

- 基于 Python 的控制系统：[python-control/python-control: The Python Control Systems Library is a Python module that implements basic operations for analysis and design of feedback control systems.](https://github.com/python-control/python-control) 📖 [API文档](https://python-control.readthedocs.io/en/latest/intro.html)

#### 🌀 基于深度学习的自适应滤波相关仓库：
- Meta-AF: 基于Meta-Learning的自适应滤波仓库：[adobe-research/MetaAF: Control adaptive filters with neural networks.](https://github.com/adobe-research/MetaAF)
- Auto-DSP：基于深度学习的声学回声消除器优化仓库：[jmcasebeer/autodsp: Train custom adaptive filter optimizers without hand tuning or extra labels.](https://github.com/jmcasebeer/autodsp)


### 📚 帮助文档：
- pyroomacoustics房间音频处理包帮助手册链接：[Contributing — Pyroomacoustics 0.6.0 documentation](https://pyroomacoustics.readthedocs.io/en/pypi-release/contributing.html)

- python_speech_features包帮助文档链接：[Welcome to python_speech_features’s documentation! — python_speech_features 0.1.0 documentation](https://python-speech-features.readthedocs.io/en/latest/)

- 音乐音频python分析包案例链接：[Advanced examples — librosa 0.9.2 documentation](https://librosa.org/doc/latest/advanced.html)
- 音乐音频python分析包帮助文档链接：[librosa — librosa 0.9.2 documentation](https://librosa.org/doc/latest/index.html)

- （FaceBook）前沿序列建模工具包（Fairseq）帮助文档链接：[fairseq documentation — fairseq 0.12.2 documentation](https://fairseq.readthedocs.io/en/latest/index.html)


### 📰 相关论文链接
- [Sparse R-CNN: End-to-End Object Detection with Learnable Proposals](https://arxiv.org/pdf/2011.12450.pdf)
- SincNet：[[1808.00158] Speaker Recognition from Raw Waveform with SincNet](https://arxiv.org/abs/1808.00158)

#### WavEncoder仓库相关的文献：
- `SoftAttention`类的实现论文：[Attention-based End-to-End Models for Small-Footprint Keyword Spotting](https://arxiv.org/pdf/1803.10916.pdf)。Github仓库：[isadrtdinov/kws-attention: Attention-based model for keywords spotting](https://github.com/isadrtdinov/kws-attention)

#### 基于深度学习的自适应滤波相关文献：
- Meta-AF: 基于Meta-Learning的自适应滤波文献：[[2204.11942] Meta-AF: Meta-Learning for Adaptive Filters](https://arxiv.org/abs/2204.11942)
- Auto-DSP：基于深度学习的声学回声消除器优化文献：[[2110.04284] Auto-DSP: Learning to Optimize Acoustic Echo Cancellers](https://arxiv.org/abs/2110.04284)

💿 **音频数据集**：

- TIMIT音频数据集：[TIMIT Corpus — Pyroomacoustics 0.6.0 documentation](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.datasets.timit.html#the-timit-dataset)

- 回音消除研究数据集（AEC-Challenge）：[microsoft/AEC-Challenge: AEC Challenge](https://github.com/microsoft/AEC-Challenge) [相关论文：ICASSP 2021 Acoustic Echo Cancellation Challenge: Datasets, Testing Framework, and Results](https://arxiv.org/pdf/2009.04972.pdf)