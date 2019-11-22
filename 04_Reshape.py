#이런식으로 데이터를 Reshape 했을 때, 학습을 하면 일어나는 차이가 큰 차이일까?

import numpy as np

x1 = np.array([
    [[1,2,3,4], [2,3,4,5], [3,4,5,6], [7,8,9,10]],
[[1,2,3,4], [2,3,4,5], [3,4,5,6], [7,8,9,10]],
[[1,2,3,4], [2,3,4,5], [3,4,5,6], [7,8,9,10]]
]
)

print(x1)
print(x1.shape)

x2 = x1.reshape(3, 4 * 4)
print(x2)
print(x2.shape)
