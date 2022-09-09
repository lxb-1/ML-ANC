def get_conv_model():
	model = Sequential()
	model.add(Conv2D(16, (3, 3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1), padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='same'))
	model.add(MaxPool2D((2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

	return model

def get_recurrent_model():
	#Shape of data for LSTM is (n, time, feature)
	model = Sequential()
	model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
	model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(0.5))
	model.add(TimeDistributed(Dense(64, activation='relu')))
	model.add(TimeDistributed(Dense(32, activation='relu')))
	model.add(TimeDistributed(Dense(16, activation='relu')))
	model.add(TimeDistributed(Dense(8, activation='relu')))
	model.add(Flatten())
	model.add(Dense(10, activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

	return model