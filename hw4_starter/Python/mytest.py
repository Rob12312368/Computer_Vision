


import numpy as np

x = np.array([[1,2],[3,-2]])
print(np.where((x[:,0] > 0) & (x[:,0] < 2)))

y = np.array([1,2,3])
print(np.where(y[:]>0))
