import tensorflow as tf
import keras
from keras.datasets import imdb
import numpy as np
from keras import layers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
num_words=10000)

word_index = imdb.get_word_index()
reverse_word_index = dict(
[(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
[reverse_word_index.get(i - 3, '?') for i in train_data[0]])

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

def model_select(choose):
    match choose:
        case "1":
            model = keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid")
            ])

            model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

            history = model.fit(partial_x_train,
            partial_y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_val, y_val))
            return history
        case "2":
            model = keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid")
            ])

            model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

            history = model.fit(partial_x_train,
            partial_y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_val, y_val))
            return history
        case "3":
            model = keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid")
            ])

            model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

            history = model.fit(partial_x_train,
            partial_y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_val, y_val))
            return history
        case "4":
            model = keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid")
            ])

            model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

            history = model.fit(partial_x_train,
            partial_y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_val, y_val))
            return history
        case "5":
            model = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid")
            ])

            model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

            history = model.fit(partial_x_train,
            partial_y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_val, y_val))
            return history
        case "6":
            model = keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid")
            ])

            model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

            history = model.fit(partial_x_train,
            partial_y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_val, y_val))
            return history
        case "7":
            model = keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid")
            ])

            model.compile(optimizer='rmsprop',
            loss='mse',
            metrics=['accuracy'])

            history = model.fit(partial_x_train,
            partial_y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_val, y_val))
            return history
        case "8":
            model = keras.Sequential([
            layers.Dense(16, activation="tanh"),
            layers.Dense(16, activation="tanh"),
            layers.Dense(1, activation="sigmoid")
            ])

            model.compile(optimizer='rmsprop',
            loss='mse',
            metrics=['accuracy'])

            history = model.fit(partial_x_train,
            partial_y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_val, y_val))
            return history
        case _:
            print("Unknown model type")

print(f'''
      Choose model:
      1 - 16 relu, 16 relu, 20 epochs (Default)
      2 - 16 relu, 20 epochs
      3 - 16 relu, 16 relu, 16 relu, 20 epochs
      4 - 32 relu, 32 relu, 20 epochs
      5 - 64 relu, 64 relu, 20 epochs
      6 - 8 relu, 8 relu, 20 epochs
      7 - Default, but loss function is mse
      8 - 16 tanh, 16 tanh, 20 epochs
      ''')





choose_in = input()
history = model_select(choose_in)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Loss on learning")
plt.plot(epochs, val_loss_values, "b", label="Loss on tests")
plt.title("Loss on learning and tests")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()
plt.show()

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Accuracy on learning")
plt.plot(epochs, val_acc, "b", label="Accuracy on tests")
plt.title("Accuracy on learning and tests")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()