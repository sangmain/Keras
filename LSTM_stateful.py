import numpy as np

batch_size = 7
num_epochs = 100


def split(x, split):
    sequences = []
    for i in range(x.shape[0] - split - 1):
        window = x[ i : i + split ]
        sequences.append(window)
    return sequences

def np_mul(data, batch_size):
    data = np.array(data)
    x_pred = np.zeros((batch_size, data.shape[1], data.shape[2]))
    for i in range(7):
        x_pred[i] = np.array(data)

    return x_pred


def generate_data(start, end):
    x1 = np.arange(start, end + 1)
    x2 = np.arange(start + 100, end + 100 + 1)

    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)

    x = np.concatenate((x1, x2), axis=1)
        
    data = np.array(split(x, 8))
    print(data.shape)
    np.random.seed(42)
    np.random.shuffle(data, )


    y = np.zeros((data.shape[0], 3, 3))
    x = data[:, :-3, :]
    y[:, :, :-1] = data[:, -3:, :]

    y[:, :, -1] = y[:, :, 0] + y[:, :, 1]
    return x, y


x, y = generate_data(1, 100)

x_test, y_test = generate_data(100, 110)


# x_test = x2[10:17]
# y_test = y2[10:17]


# x_valid = np_mul(x_valid, batch_size)
# y_valid = np_mul(y_valid, batch_size)

# print(x_valid.shape)
# print(y_valid.shape)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Reshape

model = Sequential()

model.add(LSTM(100, activation='relu', input_shape=(5, 2)))
model.add(Dense(512, activation='relu'))


model.add(Dense(256, activation='relu'))

model.add(Dense(100, activation='relu'))


model.add(Dense(9))
model.add(Reshape((3, 3)))
model.summary()


model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode = 'auto') # monitor loss , val_loss, val_acc /// loss나 val_loss사용하는것이 더 나음


for epoch_idx in range(num_epochs):
    print('epoch : '+ str(epoch_idx))
    model.fit(x, y, batch_size=batch_size, epochs=1, shuffle=False, callbacks = [early_stopping])

    model.reset_states()




# x_test = [[
#             [101,201],
#             [102,202],
#             [103,203],
#             [104,204],
#             [105,205]
#         ]]

# x_test = np_mul(x_test, batch_size)
print(x_test.shape)
y_pred = model.predict(x_test, batch_size=batch_size)
print(x_test)
print(y_pred)