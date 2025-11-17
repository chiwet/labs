from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
def model(epoch, relu, batch):
    model = keras.Sequential([
        layers.Dense(relu, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=epoch, batch_size=batch)
    test_digits = test_images[0:10]
    predictions = model.predict(test_digits)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"test_acc: {test_acc} ({epoch} эпох, {relu} relu, {batch} batch)")

model(5,512,128)
model(15,512,128)
model(5,1024,128)
model(5,512,256)
model(15,1024,128)