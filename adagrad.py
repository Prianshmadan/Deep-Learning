# import numpy as np

# # Define the learning rate, number of iterations, and initial parameter values
# learning_rate = 0.1
# num_iterations = 1000
# x = np.array([1, 2])
# y = np.array([2, 3])
# theta = np.random.rand(2)
# epsilion=1e-8
# # Define the cost function
# def cost_function(theta, x, y):
#     m = len(y)
#     predictions = np.square(x.dot(theta))
#     cost = (1/(2*m)) * np.sum(np.square(predictions-y))
#     return cost

# # Define the gradient function
# def gradient_function(theta, x, y):
#     m = len(y)
#     predictions = np.square(x.dot(theta))
#     gradient = (1/m) * x.T.dot((predictions-y) * x)
#     return gradient

# # Define the adaptive learning rate function
# def adaptive_learning_rate_function(theta, x, y, gradient, v,epsilion):
#     m = len(y)
#     predictions = np.square(x.dot(theta))
#     gradient_squared = np.square(gradient)
#     v = (1-1/m)*v + (1/m)*gradient_squared
#     adaptive_learning_rate = learning_rate / np.sqrt(v + epsilion)
#     return adaptive_learning_rate, v

# # Adaptive gradient descent
# v = np.zeros_like(theta)
# for i in range(num_iterations):
#     gradient = gradient_function(theta, x, y)
#     adaptive_learning_rate, v = adaptive_learning_rate_function(theta, x, y, gradient, v,epsilion)
#     theta = theta - adaptive_learning_rate * gradient

#     # Print the cost function every 100 iterations
#     if i % 100 == 0:
#         cost = cost_function(theta, x, y)
#         print(f"Iteration {i}: Cost={cost}")

# # Print the final parameters
# print("Final Parameters:")
# print(theta)







import numpy as np
import matplotlib.pyplot as plt

# Define the function to be minimized
def z(x, y):
    return x**2 + y**2

# Define the gradient function
def gradient(x, y):
    return np.array([2*x, 2*y])

# Define the initial parameters
x = 2
y = 1

learning_rate = 0.01
g_sq = np.zeros(2)
epsilion=1e-8
# Define the number of iterations
num_iterations = 1000
# Initialize lists to store the values for x, y, and z
x_list = [x]
y_list = [y]
z_list = [z(x, y)]

# Adaptive Gradient Descent
for i in range(num_iterations):
    grad = gradient(x, y)
    g_sq += np.square(grad)
    adjusted_learning_rate = learning_rate / np.sqrt(g_sq)+epsilion
    x -= adjusted_learning_rate[0] * grad[0]
    y -= adjusted_learning_rate[1] * grad[1]
    x_list.append(x)
    y_list.append(y)
    z_list.append(z(x, y))

# Plot the contour plot
delta = 0.01
x_range = np.arange(-5, 5, delta)
y_range = np.arange(-5, 5, delta)
X, Y = np.meshgrid(x_range, y_range)
Z = z(X, Y)
fig, ax = plt.subplots()
ax.contourf(X, Y, Z, 50)
ax.plot(x_list, y_list, 'r.-')
plt.show()

# Print the final result
print(f"The minimum value of z is {z_list[-1]} at x={x_list[-1]}, y={y_list[-1]}")