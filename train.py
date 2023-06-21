import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

data = pd.read_csv("A_Z Handwritten Data.csv")
x_az = data.drop('0', axis=1).values
y_az = data['0'].values
x_train = x_train.reshape(-1, 28, 28) / 255.0
x_az = x_az.reshape(-1, 28, 28) / 255.0
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
x_train = np.concatenate((x_train, x_az), axis=0)
y_train = np.concatenate((y_train, y_az), axis=0)
input_shape = x_train.shape[1:]
model = Sequential()

model.add(Flatten(input_shape=input_shape))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(36, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

model.save('handwriting_model.h5')