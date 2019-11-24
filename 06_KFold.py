from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np
import os
import tensorflow as tf

def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu')) #분류모델의 마지막은 softmax

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


    return model

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
import matplotlib.pyplot as plt

digit = X_train[2000]
# plt.imshow(digit, cmap = plt.cm.binary)
# plt.show()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

print(Y_train.shape)
print(Y_test.shape)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# print(X_train.shape)
# print(X_test.shape)

print(Y_train.shape)
print(Y_test.shape)

k = 4
num_val_samples = len(X_train) // k
all_scores = []
for i in range(k):
    print("fold count: ", i)
    val_data = X_train[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = Y_train[i * num_val_samples : (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([X_train[:i * num_val_samples], X_train[(i + 1) * num_val_samples :]], axis = 0)
    partial_train_targets= np.concatenate([Y_train[:i * num_val_samples], Y_train[(i + 1) * num_val_samples :]], axis = 0)

    model = build_model()


    model.fit(partial_train_data, partial_train_targets,
                    epochs=5, batch_size=200, verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))


