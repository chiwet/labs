import numpy as np
import matplotlib.pyplot as plt
import math
from math import log1p as ln
x = np.array([10, 14, 18, 22, 26, 30])
y = np.array([4.2, 4.5, 4.8, 5.1, 5.2, 5.4])
n = len(x)
scores = [0,0,0,0]
step = 0.2
sum_x = sum(x)
sum_x_2 = sum([i**2 for i in x])
sum_x_3 = sum([i**3 for i in x])
sum_x_4 = sum([i**4 for i in x])
sum_y = sum(y)
sum_xy = sum([x[i] * y[i] for i in range(n)])
sum_x2y = sum([x[i]**2 * y[i] for i in range(n)])
d_main = sum_x_2 * n - sum_x**2
d_a = sum_xy * n - sum_x * sum_y
d_b = sum_x_2 * sum_y - sum_xy * sum_x
a = d_a / d_main
b = d_b / d_main
print(f'y = {a}x + {b}')
for i in range(n):
    scores[0] += ((a * x[i] + b) - (y[i]))**2
fig, axs = plt.subplots(2,2)
for i in np.arange(0,x[-1]+3,step):
    axs[0,0].scatter(i, i * a + b, c='green', s=10)
for i in range(n):
    axs[0,0].scatter(x[i], y[i], c='black')
axs[0, 0].set_title(f"Линейная функция:")
axs[0, 0].set_xlabel("X")
axs[0, 0].set_ylabel("Y")
d_main = (sum_x_4 * sum_x_2 * n + 2 * sum_x_3 * sum_x * sum_x_2) - (sum_x_2**3 + sum_x**2 * sum_x_4 + n * sum_x_3**2)
d_a = (sum_x_3 * sum_x * sum_y + sum_x_2 * sum_xy * sum_x + sum_x2y * sum_x_2 * n) - (sum_x2y * sum_x**2 + sum_x_2**2 * sum_y + sum_x_3 * sum_xy * n)
d_b = (sum_x_4 * sum_x * sum_y + sum_x_2**2 * sum_xy + sum_x2y * sum_x_3 * n) - (sum_x2y * sum_x * sum_x_2 + sum_x_2 * sum_x_3 * sum_y + sum_x_4 * sum_xy * n)
d_c = (sum_x_4 * sum_x_2 * sum_y + sum_x_3 * sum_xy * sum_x_2 + sum_x2y * sum_x_3 * sum_x) - (sum_x2y * sum_x_2**2 + sum_x_3**2 * sum_y + sum_x_4 * sum_xy * sum_x)
a = d_a / d_main
b = d_b / d_main
c = d_c / d_main
for i in range(n):
    scores[1] += ((a * x[i]**2 + b * x[i] + c) - (y[i]))**2
print(f'y = {a}x^2 + {b}x + {c}')
for i in np.arange(0,x[-1]+3,step):
    axs[0,1].scatter(i, i**2 * a + b * i + c, c='green', s=10)
for i in range(n):
    axs[0,1].scatter(x[i], y[i], c='black')
axs[0,1].set_title(f"Квадратичная функция:")
axs[0,1].set_xlabel("X")
axs[0,1].set_ylabel("Y")
sum_x_ln = sum([ln(i - 1) for i in x])
sum_x_2_ln = sum([ln(i - 1)**2 for i in x])
sum_y_ln = sum([ln(i - 1) for i in y])
sum_xy_ln = sum([ln(x[i] - 1) * ln(y[i] - 1) for i in range(n)])
sum_x_y_ln = sum([x[i] * ln(y[i] - 1) for i in range(n)])
print(sum_x_ln, sum_x_2_ln, sum_y_ln, sum_xy_ln)

d_main = sum_x_2_ln * n - sum_x_ln**2
d_a = sum_xy_ln * n - sum_x_ln * sum_y_ln
d_b = sum_x_2_ln * sum_y_ln - sum_xy_ln * sum_x_ln
a = d_a / d_main
b = d_b / d_main
for i in range(n):
    scores[2] += (((math.e**b)*(x[i]**a)) - (y[i]))**2
print(f"y = {math.e**b}x^{a}")
for i in np.arange(0,x[-1]+3,step):
    axs[1,0].scatter(i, (math.e**b)*(i**a), c='green', s=10)
for i in range(n):
    axs[1,0].scatter(x[i], y[i], c='black')
axs[1,0].set_title(f"Степенная функция:")
axs[1,0].set_xlabel("X")
axs[1,0].set_ylabel("Y")

d_main = sum_x_2 * n - sum_x**2
d_a = sum_x_y_ln * n - sum_x * sum_y_ln
d_b = sum_x_2 * sum_y_ln - sum_x_y_ln * sum_x
a = d_a / d_main
b = d_b / d_main

print(f"y = {a}e^({b}x)")
for i in range(n):
    scores[3] += ((a*math.e**(b * x[i])) - (y[i]))**2
for i in np.arange(0,x[-1]+3,step):
    axs[1,1].scatter(i, a*math.e**(b * i), c='green', s=10)
for i in range(n):
    axs[1,1].scatter(x[i], y[i], c='black')

axs[1,1].set_title(f"Показательная функция:")
axs[1,1].set_xlabel("X")
axs[1,1].set_ylabel("Y")
plt.show()
for i in np.arange(0,x[-1]+3,step):
    plt.scatter(i, a*math.e**(b * i), c='red', s=10)
d_main = sum_x_2_ln * n - sum_x_ln**2
d_a = sum_xy_ln * n - sum_x_ln * sum_y_ln
d_b = sum_x_2_ln * sum_y_ln - sum_xy_ln * sum_x_ln
a = d_a / d_main
b = d_b / d_main
for i in np.arange(0,x[-1]+3,step):
    plt.scatter(i, (math.e**b)*(i**a), c='green', s=10)
d_main = (sum_x_4 * sum_x_2 * n + 2 * sum_x_3 * sum_x * sum_x_2) - (sum_x_2**3 + sum_x**2 * sum_x_4 + n * sum_x_3**2)
d_a = (sum_x_3 * sum_x * sum_y + sum_x_2 * sum_xy * sum_x + sum_x2y * sum_x_2 * n) - (sum_x2y * sum_x**2 + sum_x_2**2 * sum_y + sum_x_3 * sum_xy * n)
d_b = (sum_x_4 * sum_x * sum_y + sum_x_2**2 * sum_xy + sum_x2y * sum_x_3 * n) - (sum_x2y * sum_x * sum_x_2 + sum_x_2 * sum_x_3 * sum_y + sum_x_4 * sum_xy * n)
d_c = (sum_x_4 * sum_x_2 * sum_y + sum_x_3 * sum_xy * sum_x_2 + sum_x2y * sum_x_3 * sum_x) - (sum_x2y * sum_x_2**2 + sum_x_3**2 * sum_y + sum_x_4 * sum_xy * sum_x)
a = d_a / d_main
b = d_b / d_main
c = d_c / d_main
for i in np.arange(0,x[-1]+3,step):
    plt.scatter(i, i**2 * a + b * i + c, c='blue', s=10)
d_main = sum_x_2 * n - sum_x**2
d_a = sum_xy * n - sum_x * sum_y
d_b = sum_x_2 * sum_y - sum_xy * sum_x
a = d_a / d_main
b = d_b / d_main
for i in np.arange(0,x[-1]+3,step):
    plt.scatter(i, i * a + b, c='black', s=10)
for i in range(n):
    plt.scatter(x[i], y[i], c='gray')
ax = plt.gca()
ax.set_xlim([0, 33])
ax.set_ylim([0, 6])
plt.title(f"Результат работы метода наименьших квадратов:")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
print(f"""S(a,b):
    Линейная функция - {scores[0]}
    Квадратичная функция - {scores[1]}
    Показательная функция - {scores[3]}
    Степенная функция - {scores[2]}""")