import numpy as np
import random

x1 = []
x2 = []
# for i in range(100):
#     num = np.arange(0, 1000)
    
#     x1.append(num)
#     x2.append(num * 2)

x1 = np.arange(1, 1000)
x2 = x1 * 2

print("x1[0]: ", x1[0])
print("x2[0]: ", x2[0])

def split(x1, x2, split):
    x1_cpy = []
    x2_cpy = []
    for i in range(len(x1) - 10):
        x1_cpy.append(x1[i: i + split])
        x2_cpy.append(x2[i: i + split])

    return np.array(x1_cpy), np.array(x2_cpy)

x1, x2 = split(x1, x2, 5)
print(x1.shape)
print(x2.shape)
y = []
print(x2[0])
for i in range(len(x2)):
    y.append(x2[i, -1])
    x2[i, -1] = 0

print(x2[0])
y = np.array(y)
print(y.shape)
print(y[0])

x1 = x1.reshape(x1.shape[0], -1, x1.shape[1])
x2 = x2.reshape(x2.shape[0], -1, x2.shape[1])

x = np.concatenate((x1, x2), axis=1)
print(x[0])
print(x.shape)
print(y.shape)

x_train = x[:800]
x_test = x[800:]

y_train = y[:800]
y_test = y[800:]

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

from sklearn.utils import shuffle
x, y = shuffle(x, y, random_state=42)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(16, return_sequences=True, input_shape=(2, 5), activation='relu'))
model.add(LSTM(16, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, batch_size=10, epochs=10)

print("x_test: \n", x_test[0:5])

y_pred = model.predict(x_test)
print("prediction: \n", y_pred[0:5])

