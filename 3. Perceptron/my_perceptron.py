import math
import random
import numpy as np
import matplotlib.pyplot as plt


def my_perceptron(x, y, b=1, n_iter=math.inf):
    """
    This function executes the perceptron algorithm from section 3.1. 
    By means of the third (optional) argument it shall be distinguished whether a homogeneous linear hypothesis is to be learned.

    params:
    x ->      (d, m)-Matrix consisting of the m training features in R^d
    y ->      (1, m)-Vector consisting of the m associated labels {-1, +1}
    b ->      Optional argument that learns a homogeneous linear hypothesis from the data for the value 0, otherwise a general linear   hypothesis
    n_iter -> Maximum number of interations for the algorithm (by default infinity)

    return:
    w ->      Column vector containing the learned weights and bias in the form (w_1, w_2, ... w_d, b)^T
    T ->      Integer of the number of executed steps in the algorithm
    ws ->     Matrix with T+1 columns, the t-th column contains the t-th step Iterated of the procedure
    RSs ->    Row vector containing the empirical risk for each vector ws

    """

    # Reading the dimension d and the data number m from x and y, respectively.
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

        # Calculation of the obtained empirical risk
        def RS(w): return np.mean(check(w, x, y) <= 0)

        # Empirical risk of the current w:
        RSs[0] = RS(w)

        # Iterations via while loop
        t = 0
        while np.min(check(w, x, y)) <= 0 and t < n_iter:
            # Find all unsatisfied constraints
            inds = [i for (i, val) in enumerate(check(w, x, y)[0]) if val <= 0]

            # Select an unfulfilled constraint
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

         # Calculation of the obtained empirical risk
        def RS(w): return np.mean(check(w, x, y) <= 0)

        # Empirical risk of the current w:
        RSs[0] = RS(w)

        # Iterations via while loop
        t = 0
        while np.min(check(w, x, y)) <= 0 and t < n_iter:
            # Find all unsatisfied constraints
            # print(f'CHECK {check(w, x, y)}')
            inds = [i for (i, val) in enumerate(check(w, x, y)[0]) if val <= 0]
            # print(inds)

            # Select an unfulfilled constraint
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