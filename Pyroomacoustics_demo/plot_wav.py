# -*- coding: utf-8 -*-

import wave
import pylab as pl
import numpy as np

print('working')

# 打开wav文件
file = wave.open(r"/home/liang/文档/Coding/Machine Learning/Pyroomacoustics_demo/google_speech_commands/bird/0a7c2a8d_nohash_0.wav")

# 读取格式信息
params = file.getparams()
print(dir(params))

nchannels, sampwidth, framerate, nframes = params[:4]

# 读取波形数据
str_data = file.readframes(nframes)
# 文件读取完毕，关闭文件
file.close()

# 将波形数据转换为数组
# wave_data = np.fromstring(str_data, dtype=np.short)
wave_data = np.frombuffer(str_data, dtype=np.short)
# wave_data.shape = (-1, 2)

wave_data = wave_data.T     # 矩阵转置
# 根据 `n_frames` 与 `framerate` 参数转换数据的时间轴
time = np.arange(0, nframes) * (1.0 / framerate)

# 查看时间数据和波形数据
print("Time:", len(time))
print("Wave_data", len(wave_data[0:len(time)]))

# 绘制波形
"""
subplot(mnp) / (m,n,p)是将多个图画到一个平面上的工具.
其中，m表示是图排成m行，n表示图排成n列，也就是整个figure中有n个图是排成一行的，一共m行，
如果m=2就是表示2行图.p表示图所在的位置，p=1表示从左到右从上到下的第一个位置.
"""
pl.subplot(2, 1, 1)  # 这里也可以使用pl.subplot(211)
pl.plot(time, wave_data[0:len(time)])
pl.subplot(2, 1, 2)  # 这里也可以使用pl.subplot(212)

pl.plot(time, wave_data[0:len(time)], c="g")
pl.xlabel("time (seconds)")
pl.show()