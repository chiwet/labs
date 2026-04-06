import tensorflow as tf
import keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
from keras import layers

#Models
model1 = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="tanh"),
    layers.Dense(10, activation="softmax")
])

model2 = keras.Sequential([
    layers.Dense(256, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model3 = keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model4 = keras.Sequential([
    layers.Dense(128, activation="sigmoid"),
    layers.Dense(10, activation="softmax")
])

model5 = keras.Sequential([
    layers.Dense(512, activation="tanh"),
    layers.Dense(512, activation="tanh"),
    layers.Dense(10, activation="softmax")
])

model1.compile(optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
model2.compile(optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
model3.compile(optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
model4.compile(optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

model5.compile(optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

model1.fit(train_images, train_labels, epochs=8, batch_size=512)
test_loss1, test_acc1 = model1.evaluate(test_images, test_labels)
print(f"test_acc 1: {test_acc1}")

model2.fit(train_images, train_labels, epochs=5, batch_size=512)
test_loss2, test_acc2 = model2.evaluate(test_images, test_labels)
print(f"test_acc 2: {test_acc2}")

model3.fit(train_images, train_labels, epochs=3, batch_size=128)
test_loss3, test_acc3 = model3.evaluate(test_images, test_labels)
print(f"test_acc 3: {test_acc3}")

model4.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss4, test_acc4 = model4.evaluate(test_images, test_labels)
print(f"test_acc 4: {test_acc4}")

model5.fit(train_images, train_labels, epochs=10, batch_size=256)
test_loss5, test_acc5 = model5.evaluate(test_images, test_labels)
print(f"test_acc 5: {test_acc5}")

print(f""" 
      Model 1 acc: {test_acc1}\n
      Model 2 acc: {test_acc2}\n
      Model 3 acc: {test_acc3}\n
      Model 4 acc: {test_acc4}\n
      Model 5 acc: {test_acc5}\n""")