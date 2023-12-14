import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

data = keras.datasets.mnist

(train_images, train_labels),(test_images,test_labels) = data.load_data()

train_images = train_images /255
test_images = test_images / 255

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10,activation = "softmax"))

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']);

model.fit(train_images,train_labels, epochs=3);

test_loss, test_accuracy = model.evaluate(test_images,test_labels)

print("Preciznost:",test_accuracy)


plt.imshow(train_images[3], cmap=plt.cm.binary);
plt.title(train_labels[3])
plt.show()