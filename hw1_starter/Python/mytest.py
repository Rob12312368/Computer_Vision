import matplotlib.pyplot as plt 
import random
import numpy as np


second = []
j = 0.75
zeros = 0
ones = 0
for i in range(1000):
    tmp = random.random()
    if tmp <= j:
        ones += 1
    else:
        zeros += 1
print(zeros, ones)
plt.bar([0, 1], [zeros, ones], width = 0.1, color = 'blue', edgecolor='black')
plt.show()
