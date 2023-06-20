import tensorflow as tf
# import numpy as np

# # Define the hyperparameters
# learning_rate = 0.01
# num_epochs = 100
# batch_size = 32
# input_size = 5
# hidden_size = 4

# # Define the input data



# # Define the autoencoder model
# inputs = tf.keras.Input(shape=(input_size,))
# encoded = tf.keras.layers.Dense(hidden_size, activation='relu')(inputs)
# decoded = tf.keras.layers.Dense(input_size, activation='linear')(encoded)
# autoencoder = tf.keras.Model(inputs, decoded)

# # Compile the model
# autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#                     loss='mse')


# autoencoder.fit(x_train, x_train, epochs=num_epochs, batch_size=batch_size)


# decoded_output = autoencoder.predict(x_train)

# print("Input data:")
# print(x_train)
# print("Reconstructed data:")
# print(decoded_output)

