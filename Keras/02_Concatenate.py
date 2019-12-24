# 서로 크기가 다른 데이터를 Concatenate 하면 합쳐지는데 이게 학습이 될만 하도록 Concatenate 된걸까?
import numpy as np

x1 = np.array([ [1,2,3,4,5],
         [2,3,4,5,6],
         [3,4,5,6,7]])

x2 = np.array([ [1,2,3,4],
        [2,3,4,5,],
        [3,4,5,6]])

print(x1.shape)
print(x2.shape)
x3 = np.concatenate((x1, x2), axis=1)

print(x3)
print(x3.shape)