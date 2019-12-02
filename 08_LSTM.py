<<<<<<< HEAD
##### 1~100의 숫자 시퀀스로 101~105의 시퀀스를 예측한다

=======
>>>>>>> 6637b3519b923a6ba26d35f182294453b94227f4
import numpy as np

x1 = np.arange(1, 100 + 1)
x2 = np.arange(101, 200 + 1)

x1 = np.expand_dims(x1, axis=1)
x2 = np.expand_dims(x2, axis=1)

x = np.concatenate((x1, x2), axis=1)
def split(x, split):
    sequences = []
    for i in range(x.shape[0] - split - 1):
        window = x[ i : i + split ]
        sequences.append(window)
        print(window.shape)

    return sequences
data = np.array(split(x, 8))
<<<<<<< HEAD
<<<<<<< HEAD
np.random.shuffle(data)

=======

import random
random.shuffle(data)
>>>>>>> 6637b3519b923a6ba26d35f182294453b94227f4
=======

import random
random.shuffle(data)
>>>>>>> 6637b3519b923a6ba26d35f182294453b94227f4
print(data.shape)

y = np.zeros((data.shape[0], 3, 3))
x = data[:, :-3, :]
y[:, :, :-1] = data[:, -3:, :]

print(x.shape)
print(y.shape)

y[:, :, -1] = y[:, :, 0] + y[:, :, 1]

<<<<<<< HEAD

print(x[:2])
print(y[:2])


x_valid = [[
            [93,193],
            [94,194],
            [95, 195],
            [96, 196],
            [97,197]
        ]]

y_valid = [[
            [98, 198, 296],
            [99, 199, 297],
            [100, 200, 298]
            ]]


x_valid = np.array(x_valid)
y_valid = np.array(y_valid)

print(x_valid.shape)
print(y_valid.shape)

=======
>>>>>>> 6637b3519b923a6ba26d35f182294453b94227f4
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Reshape

model = Sequential()

<<<<<<< HEAD
model.add(LSTM(16, activation='relu', input_shape=(5, 2)))
model.add(Dense(512, activation='relu'))


model.add(Dense(256, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(9))
model.add(Reshape((3, 3)))
model.summary()
=======
model.add(LSTM(16, return_sequences=True, input_shape=(5, 2)))
model.add(LSTM(16))

model.add(Dense(9, activation='relu'))
model.summary()
model.add(Reshape((3, 3)))
>>>>>>> 6637b3519b923a6ba26d35f182294453b94227f4


model.compile(optimizer='adam', loss='mse')

<<<<<<< HEAD
model.fit(x, y, batch_size=10, epochs=100, validation_data=(x_valid, y_valid))

=======
model.fit(x, y, batch_size=5, epochs=100)
>>>>>>> 6637b3519b923a6ba26d35f182294453b94227f4

x_test = [[
            [101,201],
            [102,202],
            [103,203],
            [104,204],
            [105,205]
        ]]

x_test = np.array(x_test)
<<<<<<< HEAD
print(x_test)
y_pred = model.predict(x_test)
print(y_pred)
=======
print(x_test.shape)
model.predict(x_test)
>>>>>>> 6637b3519b923a6ba26d35f182294453b94227f4
