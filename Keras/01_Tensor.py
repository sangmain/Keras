import numpy as np

#### scalar, 0D Tensor
x = np.array(12)
print(x.shape)
print(x.ndim)

#### Vector, 1D Tensor
x = np.array([12])
print(x.shape)
print(x.ndim)

#### Matrix, 2D Tensor
x = np.array([[12]])
print(x.shape)
print(x.ndim)

#### 3D Tensor
x = np.array([ [ [1,2,3], [2,3,4] ],
      [ [1,2,3], [2,3,4] ]
])
print(x.shape)
print(x.ndim)
