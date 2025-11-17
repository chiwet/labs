import matplotlib.pyplot as plt
import numpy as np
import random as rand
from math import sqrt
rand.seed(a=None)
a = np.random.rand(100,2)
a = np.concatenate((a, np.zeros((100,1))), axis=1)
colors = ["#df2929", "#fd7e0e", "#4527f1", "#ff35c2", "#00ffd5"]
k = rand.randint(3,5)
b = np.random.rand(k,2)

def clusterisation(a, b):
    for i in range(len(a)):
        min_d = 99999
        nearest_cluster_i = 0
        for j in range(len(b)):
            distance = sqrt((a[i, 0] - b[j, 0])**2+(a[i, 1] - b[j, 1])**2)
            if distance < min_d:
                min_d = distance
                nearest_cluster_i = j
        a[i, 2] = nearest_cluster_i


def centralization(a, b):
    for i in range(k):
        avg_cord = [0, 0]
        amount = 0
        for j in range(len(a)):
            if a[j, 2] == i:
                avg_cord[0] += float(a[j, 0])
                avg_cord[1] += float(a[j, 1])
                amount += 1
        avg_cord[0] = avg_cord[0] / amount
        avg_cord[1] = avg_cord[1] / amount
        # print(avg_cord)
        b[i] = [avg_cord[0], avg_cord[1]]

def draw(amount):
    for i in range(k):
        plt.scatter(b[i, 0], b[i, 1], c=colors[i], s=75)
    for i in range(100):
        plt.scatter(a[i, 0], a[i, 1], c=colors[int(a[i, 2])])


    plt.title(f"Результаты кластеризации методом k-mean: итерация {amount+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()





for i in range(int(input("Введите количество итераций"))):
    clusterisation(a, b)
    centralization(a, b)
    if i == 0:
        draw(i)
draw(i)