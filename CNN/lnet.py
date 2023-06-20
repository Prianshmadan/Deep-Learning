import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Define the input shape
input_shape = x_train[0].shape

# Define the model architecture
input_img = Input(shape=input_shape)
conv1 = Conv2D(96, (3, 3), activation='relu', padding='same')(input_img)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(192, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D((2, 2))(conv2)
conv3 = Conv2D(192, (3, 3), activation='relu', padding='same')(pool2)
conv4 = Conv2D(192, (3, 3), activation='relu', padding='same')(conv3)
conv5 = Conv2D(192, (3, 3), activation='relu', padding='same')(conv4)
pool5 = MaxPooling2D((2, 2))(conv5)
flatten = Flatten()(pool5)
fc1 = Dense(3072, activation='relu')(flatten)
dropout1 = Dropout(0.5)(fc1)
fc2 = Dense(4096, activation='relu')(dropout1)
dropout2 = Dropout(0.5)(fc2)
output = Dense(10, activation='softmax')(dropout2)

# Create the model
model = Model(inputs=input_img, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train,
          epochs=10,
          batch_size=256,
          validation_data=(x_test, y_test))
ṇ