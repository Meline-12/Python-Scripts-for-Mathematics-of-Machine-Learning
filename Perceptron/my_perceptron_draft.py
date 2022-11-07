import math
import random
import numpy as np
import matplotlib.pyplot as plt


def my_perceptron(x, y, b=1, n_iter=math.inf):
    """
    This function executes the perceptron algorithm from section 3.1. 
    The third (optional) argument is used to distinguish whether a homogeneous linear hypothesis is learned or not.

    params:
    x ->      (d, m)-Matrix consisting of the m training features in R^d
    y ->      (1, m)-Vector consisting of the m associated labels {-1, +1}
    b ->      Optional argument that learns a homogeneous linear hypothesis from the data for the value 0, otherwise a general linear hypothesis
    n_iter -> Maximum number of interations for the algorithm (by default infinity)

    return:
    w ->      Column vector containing the learned weights and biases in the form (w_1, w_2, ... w_d, b)^T
    T ->      Integer of the number of executed steps in the algorithm
    ws ->     Matrix with T+1 columns, the t-th column contains the t-th iteration the process
    RSs ->    Row vector containing the empirical risk for each vector ws

    """

    # Read the dimension d from x and the number of data m from y
    d = np.size(x, axis=0)
    m = np.size(y, axis=1)

    RSs = np.zeros((1, 1))

    # Case discrimination, whether homogeneous hypothesis should be learned
    if b == 1:
        # The case of the general affine-linear hypothesis

        # Initialize extended weight vector
        w = np.zeros((d+1, 1))

        # First entry in ws:
        ws = w

        # Function to check the constraints
        def check(w, x, y): return np.multiply(y, np.dot(w.T, np.append(x, np.ones((1, m)), axis=0)))

        # Calculation of the achieved empirical risk
        def RS(w): return np.mean(check(w, x, y) <= 0)

        # Empirical risk of the current w:
        RSs[0] = RS(w)

        # Iterations over while loop
        t = 0
        while np.min(check(w, x, y)) <= 0 and t < n_iter:
            # Find all unsatisfied constraints
            inds = [i for (i, val) in enumerate(check(w, x, y)[0]) if val <= 0]

            # Select a constraint that is not fulfilled
            i = random.choice(inds)

            # Update according to iteration rule
            v = x[:, i] 
            w = w + y[0][i] * np.append(v[:, None], np.ones((1, 1)), axis=0)

            # Save current w in ws
            ws = np.append(ws, w, axis=1)

            # Calculate empirical risk and store in RSs
            RSs = np.append(RSs, RS(w))

            # Increase step counter
            t += 1

    # The case of the homogeneous linear hypothesis with b = 0
    else:
       
        # Initialize extended weight vector
        w = np.zeros((d, 1))

        # First entry in ws:
        ws = w

         # Function to check the constraints (without b)
        def check(w, x, y): return np.multiply(y, np.dot(w.T, x))

         # Calculation of the achieved empirical risk
        def RS(w): return np.mean(check(w, x, y) <= 0)

        # Empirical risk of the current w:
        RSs[0] = RS(w)

        # Iterations over while loop
        t = 0
        while np.min(check(w, x, y)) <= 0 and t < n_iter:
            # Find all unsatisfied constraints
            # print(f'CHECK {check(w, x, y)}')
            inds = [i for (i, val) in enumerate(check(w, x, y)[0]) if val <= 0]
            # print(inds)

            # Select a constraint that is not fulfilled
            i = random.choice(inds)
            # print(i)

            # Update according to iteration rule
            v = x[:, i]
            w = w + y[0][i] * v[:, None]

            # Save current w in ws
            ws = np.append(ws, w, axis=1)

            # Calculate empirical risk and store in RSs
            RSs = np.append(RSs, RS(w))
            # RSs[t+1] = RS(w)

            # Increase step counter
            t += 1
    
    # step count as output
    T = t

    return [w, T, ws, RSs]


# # Preparation

# # Generate the training data

# # Size of the dataset
# m = 25
# x = np.random.uniform(low=-3, high=3, size=(2, m))
# # print(x)

# # wahre trennende Hyperebene
# w_true = np.array([[1], [2]])
# # print(w_true)

# # The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
# y = np.sign(np.dot(w_true.T, x)) + (np.dot(w_true.T, x) == 0)
# # print(y)

# # Graphical visualization of the training data

# # First plot the true hyperplane for x in [-3,3].
# fig, ax = plt.subplots()
# ax.plot([-3,3], -w_true[0]/w_true[1]*[-3,3], "--", label="true hyperplane")
# plt.xlabel("x1")
# plt.ylabel("x2")

# # Then enter the classified points

# # Points with mark 1
# inds = [i for (i, val) in enumerate(y[0]) if val == 1]
# ax.scatter(x[0][inds], x[1][inds], c="b", marker="+", linewidths = 2)
# # print(inds)
# # print(x[0][inds])
# # print(x[1][inds])

# # Points with mark -1
# indm = [i for (i, val) in enumerate(y[0]) if val == -1]
# ax.scatter(x[0][indm], x[1][indm], c="r", marker="d", linewidths = 2)
# # print(indm)


# # Perceptron algorithm

# # Apply the algorithm to the data with b = 0
# [w, T, ws, RSs] = my_perceptron(x, y, 1, 100)
# print(f'T = {T}')
# # print([w, T, ws, RSs])

# # Plot the learned hypothesis on the graph
# ax.plot([-3,3], -w[0]/w[1]*[-3,3], "g", label="learned hypothesis")


# ax.set(xlim=(-3, 3), ylim=(-3, 3))
# ax.axis('equal')
# fig.tight_layout()
# plt.legend()
# plt.show()


# w = np.zeros((1, 2))
# print(w)
# c = np.matrix([[1, 2], [3, 4]])
# e = np.matrix([[5, 6], [3, 4]])
# f = np.concatenate([c[1, :], np.ones((1, 1))], axis=1)
# f = np.append(c, w, axis=0)
# f = np.dot(c, e)
# print(f)
# d = np.append(c, w, axis=1)
# print(d)

