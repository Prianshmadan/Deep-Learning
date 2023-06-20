import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def binary_crossentropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

np.random.seed(42)


input_size = 5
hidden_size = 4
output_size = 5

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


learning_rate = 0.01
num_epochs = 1000


X = np.array([[1, 0, 0, 1,1]])

# Train the autoencoder
for epoch in range(num_epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y = sigmoid(z2)
    
    # Backward pass
    d_z2 = y - X
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(z1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    
    
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    
    
    loss = binary_crossentropy(X,y)
    
    
#  epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss}")




# import tensorflow as tf
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

