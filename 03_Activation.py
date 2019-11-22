import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))
def ReLU(x):
    return np.maximum(0,x)
x = np.array([-5, 10, 1.,2.,3.,4.,5., 100])

print("X data: ", x)
y = sigmoid(x)
# y = ReLU(x)
print("After activation: ", y)

