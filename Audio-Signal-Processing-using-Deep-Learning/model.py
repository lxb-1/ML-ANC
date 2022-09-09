import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM, Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
from cfg import Config
from model_architecture import get_conv_model, get_recurrent_model

def check_data():
	if os.path.isfile(config.p_path):
		print('Loading existing data for {} model'.format(config.mode))
		with open(config.p_path, 'rb') as handle:
			tmp = pickle.load(handle)
			return tmp
	else:
		return None

def build_rand_feat():
	#Creating Samples
	X, y = [], []
	_min, _max = float('inf'), -float('inf')
	for _ in tqdm(range(n_samples)):
		rand_class = np.random.choice(class_dist.index, p=prob_dist)
		file = np.random.choice(df[df.Label==rand_class].index)
		rate, wav = wavfile.read(file)
		label = df.at[file, 'Label']
		rand_index = np.random.randint(0, wav.shape[0] - config.step)
		sample = wav[rand_index:rand_index + config.step]
		X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
		_min = min(np.amin(X_sample), _min)
		_max = max(np.amax(X_sample), _max)
		X.append(X_sample)
		y.append(classes.index(label))
	config.min = _min
	config.max = _max
	X, y = np.array(X), np.array(y)
	X = (X - _min) / (_max - _min)
	if config.mode == 'conv':
		X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
	elif config.mode == 'time':
		X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
	y = to_categorical(y, num_classes=10)

	return X, y


df = pd.read_csv("/content/drive/MyDrive/Audio Signal Processing/Clean Data/instruments_clean.csv")
df.set_index('Index', inplace=True)

classes = list(np.unique(df.Label))
class_dist = df.groupby(['Label'])['length'].mean()

n_samples = 2*int(df['length'].sum()/0.1)
prob_dist = class_dist/class_dist.sum() #Probability Distribution
choices = np.random.choice(class_dist.index, p=prob_dist)

config = Config(mode='conv')
if config.mode == 'conv':
	X, y = build_rand_feat()
	y_flat = np.argmax(y, axis=1)
	input_shape = (X.shape[1], X.shape[2], 1)
	model = get_conv_model()

elif config.mode == 'time':
	X, y = build_rand_feat()
	y_flat = np.argmax(y, axis=1)
	input_shape = (X.shape[1], X.shape[2])
	model = get_recurrent_model()

class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
model.fit(X, y, epochs=10, batch_size=32, shuffle=True, class_weight=class_weight)

