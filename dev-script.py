import math

import numpy as np

a = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1]])
print(a.shape)

b = np.array([[1], [1], [1], [1]])
print(b.shape)

c = a*b.T

print(c)

L = 1*math.log(0.9)

print(L)