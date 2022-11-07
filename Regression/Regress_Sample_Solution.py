import numpy as np
import matplotlib.pyplot as plt


# Input data
x = np.array([[0.01], [0.13], [0.37], [0.4], [0.85], [0.7], [0.77], [0.84], [0.87], [0.9]])          # as a column vector
y = np.array([[0.58], [1.01], [0.15], [0.43], [0.23], [-0.23], [-0.69], [0.27], [-0.96], [-0.08]])

z = np.array([[i] for i in np.arange(0, 1 + 0.01, 0.01)])
q = np.array([[i] for i in range(0, 2)])


# Define regression functions
def f1(x): return np.ones((len(x), 1))
def f2(x): return x
def f3(x): return np.array([i**2 for i in x])
def f4(x): return np.array([i**3 for i in x])
def f5(x): return np.array([i**4 for i in x])
def f6(x): return np.array([i**5 for i in x])
def f7(x): return np.array([i**6 for i in x])
def f8(x): return np.array([i**7 for i in x])
def f9(x): return np.array([i**8 for i in x])


# Create design matrix F
F = np.concatenate([f1(x), f2(x)], axis=1)


# Sub-task e)

# Solve curve fitting problem
w_S = np.linalg.solve(np.matmul(F.T, F), np.matmul(F.T, y))
print(f"w = {w_S}")


# Calculate empirical risk of h_S
def h_S(x): return w_S[0] + w_S[1]*x

R_S = np.mean((y - h_S(x))**2)
print(f"R_S = {R_S}")


# Define true hypothesis
def h_true(x): return 1-2*x


# Calculate empirical risk of the true hypothesis
R_S_true = np.mean((y - h_true(x))**2)
print(f"R_S_true = {R_S_true}")


# Graphical visualization
fig, ax = plt.subplots()
ax.plot(q, h_S(q), "b", label="learned hypothesis h_S")
ax.plot(q, h_true(q), "r", label="true hypothesis h_true")
ax.scatter(x, y, label="Data points")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")


# Sub-task i)

# Generate vector with 1000 uniformly distributed random numbers
x_new = np.random.uniform(size=(1000, 1))
# print(f"x_new = {x_new}")


# Generate vector with 1000 normally distributed random numbers
y_new = h_true(x_new) + np.random.normal(size=(1000, 1))
# print(f"y_new = {y_new}")

# Create design matrix F
F = np.concatenate([f1(x), f2(x), f3(x), f4(x), f5(x), f6(x), f7(x), f8(x), f9(x)], axis=1)
print(f"F = {F}")

# Solve curve fitting problem
w_S = np.linalg.solve(np.matmul(F.T, F), np.matmul(F.T, y))
print(f"w = {w_S}")

# Calculate empirical risk of h_S_new
def h_S_new(x): return w_S[0]*f1(x) + w_S[1]*f2(x) + w_S[2]*f3(x) + w_S[3] * \
    f4(x) + w_S[4]*f5(x) + w_S[5]*f6(x) + w_S[6]*f7(x) + w_S[7]*f8(x) + w_S[8]*f9(x)


R_S_new = np.mean((y - h_S_new(x))**2)
print(f"R_S_new = {R_S_new}")


# Graphical visualization
fig2, ax2 = plt.subplots()
ax2.plot(z, h_S_new(z), label="learned hypothesis h_S_new")
ax2.scatter(x, y, label="Data points")
plt.grid(True)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")


# Estimate expected risk
R_D = np.mean((y_new - h_S(x_new))**2)
R_D_neu = np.mean((y_new - h_S_new(x_new))**2)
print(f"R_D = {R_D}")
print(f"R_D_neu = {R_D_neu}")

plt.show()
