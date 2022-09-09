import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa


def plot_signals(signals):
	"""绘制信号图

	Args:
		signals (_type_): _description_
	"""

	# 绘制一个2行5列的阵列图
	fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))

	# 下面分别绘制子图
	fig.suptitle('Time Series', size=16)
	i = 0
	for x in range(2):
		for y in range(5):
			axes[x,y].set_title(list(signals.keys())[i])
			axes[x,y].plot(list(signals.values())[i])
			axes[x,y].get_xaxis().set_visible(False)
			axes[x,y].get_yaxis().set_visible(False)
			i += 1

def plot_fft(fft):
	"""绘制傅立叶变换结果

	Args:
		fft (_type_): _description_
	"""
	# 绘制一个2行5列的阵列图
	fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
	# 下面分别绘制子图
	fig.suptitle('Fourier Transforms', size=16)
	i = 0
	for x in range(2):
		for y in range(5):
			data = list(fft.values())[i]
			Y, freq = data[0], data[1]
			axes[x,y].set_title(list(fft.keys())[i])
			axes[x,y].plot(freq, Y)
			axes[x, y].get_xaxis().set_visible(False)
			axes[x,y].get_yaxis().set_visible(False)
			i += 1

def plot_fbank(fbank):
	"""绘制滤波器组系数

	Args:
		fbank (_type_): _description_
	"""
	fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
	fig.suptitle('Filter Bank Coefficients', size=16)
	i = 0
	for x in range(2):
		for y in range(5):
			axes[x,y].set_title(list(fbank.keys())[i])
			axes[x,y].imshow(list(fbank.values())[i], cmap='hot', interpolation='nearest')
			axes[x,y].get_xaxis().set_visible(False)
			axes[x,y].get_yaxis().set_visible(False)
			i += 1

def plot_mfccs(mfccs):
	"""绘制梅尔频率倒谱系数(Mel Frequency Cepstrum Coefficients, MFCCs)

	Args:
		mfccs (_type_): _description_
	"""
	fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
	fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
	i = 0
	for x in range(2):
		for y in range(5):
			axes[x,y].set_title(list(mfccs.keys())[i])
			axes[x,y].imshow(list(mfccs.values())[i], cmap='hot', interpolation='nearest')
			axes[x,y].get_xaxis().set_visible(False)
			axes[x,y].get_yaxis().set_visible(False)
			i += 1

def calc_fft(signal, rate):
	"""计算离散信号的傅立叶变换

	Args:
		signal (_type_): _description_
		rate (_type_): _description_

	Returns:
		Y (_type_): _description_
		freq (_type_): DFT的采样频率
	"""
	n = len(signal)
	# 返回离散傅立叶变换DFT(Discrete Fourier Transform)的采样频率(通常结合rfft, irfft使用).
	freq = np.fft.rfftfreq(n, d=1/rate)
	# 计算1D离散傅立叶变换的实数部分
	Y = abs(np.fft.rfft(signal)/n)
	return (Y, freq)

def envelope(signal, rate, threshold):
	#Get rid of the low magnitude signal from the signal
	"""去除信号中的低幅信号

	Args:
		signal (_type_): _description_
		rate (_type_): _description_
		threshold (_type_): 阈值

	Returns:
		mask (列表): 剔除低幅数据点后的信号
	"""
	mask = []
	# 将输入信号signal转换pd.Series格式的数据，并取其绝对值
	y = pd.Series(signal).apply(np.abs)
	# 使用滑动窗口函数rolling()，求取以当前数据点左右总共rate/10个数据的平均值，将其保存为y_mean
	y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
	# 如果当前数据的y_mean小于阈值threshod，则剔除当前数据。
	for mean in y_mean:
		if mean > threshold:
			mask.append(True)
		else:
			mask.append(False)
	return mask

# Loading the data
# df = pd.read_csv("/content/drive/MyDrive/Audio Signal Processing/Data/instruments.csv")
df = pd.read_csv("Audio-Signal-Processing-using-Deep-Learning/Data/pd_babble.csv")
# df = df.drop(columns='Unnamed: 0')
# df.set_index('Index', inplace=True)
print(df)

for f in tqdm(df.index):
	rate, signal = wavfile.read(f)
	df.at[f, 'length'] = signal.shape[0]/rate #This will give us the length of the signal

classes = list(np.unique(df.Label))
class_dist = df.groupby(['Label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y = 1.08)
ax.pie(class_dist, labels = class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)

#Visualizing the data
signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
	wav_file = df[df.Label == c].iloc[0,0]
	signal, rate = librosa.load(wav_file, sr=44100)
	mask = envelope(signal, rate, 5e-4)
	signal = signal[mask]
	signals[c] = signal
	fft[c] = calc_fft(signal, rate)
	bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
	fbank[c] = bank
	mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
	mfccs[c] = mel

plot_signals(signals)
plt.show()
plot_fft(fft)
plt.show()
plot_fbank(fbank)
plt.show()
plot_mfccs(mfccs)
plt.show()

#Downsampling the audio files that we will be using for modelling
clean_data_path = '/content/drive/MyDrive/Audio Signal Processing/Clean Data'
for f in tqdm(df.Index):
	signal, rate = librosa.load(f, sr=16000)#downsample to 16000
	mask = envelope(signal, rate, 5e-4)
	path = f.split('/')[-2:]
	if os.path.exists(os.path.join(clean_data_path, path[0])) == False:
		os.mkdir(os.path.join(clean_data_path, path[0]))
	wavfile.write(filename = os.path.join(clean_data_path, path[0], path[1]),
	 rate=rate, data=signal[mask])
