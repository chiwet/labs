import numpy as np
import random as rand
l = rand.randint(0,15)
a = np.random.randint(10,size=(l,l))
b = np.array([np.random.randint(0,l),np.random.randint(0,l)])
print(a)
print(b)
def neighbors(a, b):
    c = []
    rows, cols = a.shape
    if (b[0] - 1 >= 0):
        c.append([b[0] - 1, b[1]])
    if (b[0] + 1 < rows):
        c.append([b[0] + 1, b[1]])
    if (b[1] - 1 >= 0):
        c.append([b[0], b[1] - 1])
    if (b[1] + 1 < cols):
        c.append([b[0], b[1] + 1])
    return np.array(c)
print(neighbors(a,b))
