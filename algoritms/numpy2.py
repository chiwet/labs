import numpy as np
import random as rand
r1 = rand.randint(1,8)
r2 = rand.randint(1,8)
a = np.random.randint(30,size=(r1))
b = np.random.randint(30,size=(r2))
print(a)
print(b)
c = np.zeros_like(a)
for i, a_val in enumerate(a):
    idx = np.argmin(np.abs(b - a_val))
    c[i] = a_val + b[idx]
print(c)