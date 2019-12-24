import numpy as np

import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))
def ReLU(x):
    return np.maximum(0,x)

def LeakyRelu(x): # Leaky ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
    return np.maximum(0.01*x,x) # same
x = np.array([-5, 10, 0, 1.,2.,3.,4.,5., 100])

print("X data: ", x)
y = sigmoid(x)
y = LeakyRelu(x)
# y = ReLU(x)
print("After activation: ", y)

