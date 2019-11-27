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
np.random.shuffle(data)

print(data.shape)

y = np.zeros((data.shape[0], 3, 3))
x = data[:, :-3, :]
y[:, :, :-1] = data[:, -3:, :]

print(x.shape)
print(y.shape)

y[:, :, -1] = y[:, :, 0] + y[:, :, 1]


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

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Reshape

model = Sequential()

model.add(LSTM(16, activation='relu', input_shape=(5, 2)))
model.add(Dense(512, activation='relu'))


model.add(Dense(256, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(9))
model.add(Reshape((3, 3)))
model.summary()


model.compile(optimizer='adam', loss='mse')

model.fit(x, y, batch_size=10, epochs=100, validation_data=(x_valid, y_valid))


x_test = [[
            [101,201],
            [102,202],
            [103,203],
            [104,204],
            [105,205]
        ]]

x_test = np.array(x_test)
print(x_test.shape)
y_pred = model.predict(x_test)
print(y_pred)