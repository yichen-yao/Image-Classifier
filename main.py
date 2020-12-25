import numpy as np
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path="mnist.npz")

# normalize and reshape data
train_images = train_images/255.0
test_images = test_images/255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(8, 3, input_shape=(28,28,1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
    )

# train the model
model.fit(
    train_images, 
    to_categorical(train_labels), 
    epochs=3, 
    validation_data=(test_images, to_categorical(test_labels))
    )

predictions = model.predict(test_images[:5])
print("The prediction is: ", np.argmax(predictions, axis=1))
print("The label is: ", test_labels[:5])