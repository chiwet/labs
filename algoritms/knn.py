import matplotlib.pyplot as plt
import numpy as np
import random as rand
from collections import Counter
rand.seed(a=None)
k = 5
t = 10
d1c = 50
d2c = 50
colors = ["#df2929","#4527f1"]
def euclidean_dist(x1, x2, y1, y2):
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist

def predict(X):
    predictions = [_predict(x) for x in X]
    return [int(i) for i in predictions]

def _predict(x):
    #Расчёт расстояния
    distances = [[euclidean_dist(x[0], data_x[0], x[1], data_x[1]), data_x[2]] for data_x in data1]
    distances += [[euclidean_dist(x[0], data_x[0], x[1], data_x[1]), data_x[2]] for data_x in data2]
    distances = np.array(distances)
    #Ближайшее k
    k_val = distances[np.argsort(distances[:, 0], kind='mergesort')]
    k_nearest_labels = k_val[:t,1]
    most_common = Counter(k_nearest_labels).most_common()
    return most_common[0][0]
    
data1 = np.concatenate((np.random.normal(-5,3,size=(d1c,2)), np.full((d1c,1), 1)), axis=1)
data2 = np.concatenate((np.random.normal(5,4,size=(d2c,2)), np.full((d2c,1), 2)), axis=1)
data_test = np.concatenate((np.random.normal(0,6,size=(t,2)), np.full((t,1), 0)), axis=1)
 
print(data1)
print(data2)
print(data_test)

for i in range(d1c):
    plt.scatter(data1[i, 0], data1[i, 1], c=colors[0])
for i in range(d2c):
    plt.scatter(data2[i, 0], data2[i, 1], c=colors[1])
for i in range(t):
    plt.scatter(data_test[i, 0], data_test[i, 1], c="gray", s=100)


plt.title("До использования алгоритма:")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

result = predict(data_test)

for i in range(d1c):
    plt.scatter(data1[i, 0], data1[i, 1], c=colors[0])
for i in range(d2c):
    plt.scatter(data2[i, 0], data2[i, 1], c=colors[1])
for i in range(t):
    plt.scatter(data_test[i, 0], data_test[i, 1], c=colors[result[i] - 1], s=100)


plt.title("После использования алгоритма:")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

