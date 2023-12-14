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

while True:
    index = int(input("Izaberi broj slike za testiranje (0 - 9999): "))
    if index<0 or index >9999:
        index = int(input("Izaberite novu vrednost izmedju 0 i 9999: "))

    img = test_images[index]

    img_input = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img_input)
    predicted_label = np.argmax(predictions)

    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f"Model predvidja da je ovo broj: {predicted_label}, Ovo predstavlja broj: {test_labels[index]}")
    plt.show()