import numpy as np
from matplotlib import pyplot as plt
x = np.array([1, 3.5, 6])
y = np.array([4, 5.5, 9])
l_r=0.01
num_iterations = 1000
w=0
b=0

def loss_func(w,b,x,y):
    prediction= w*x+b
    cost=np.sum((prediction-y)**2)
    return cost 


def gradient(w,b,x,y):
    prediction= w*x+b
    gradient_w=np.sum(-2*x*(y-prediction))
    gradient_b=np.sum(-2*(y-prediction)*2)
    return  gradient_w, gradient_b

for i in range(num_iterations):
    gradient_w, gradient_b=gradient(w,b,x,y)
    step_size1=l_r*gradient_w
    step_size2=l_r*gradient_b
    w= w-step_size1
    b=b-step_size2
    if i % 100 == 0:
        cost = loss_func(w, b, x, y)
        print(f"Iteration {i}: Cost={cost}, w={w}, b={b}")
plt.scatter(x, y)
plt.plot(x, w*x+b, color='red')
plt.show()