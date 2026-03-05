import numpy as np
print("NumPy version:", np.__version__)
a = np.random.rand(100, 100)
b = np.random.rand(100, 100)
c = np.dot(a, b)
print("Dot product successful")
d = np.matmul(a, b)
print("Matmul successful")
print("All basic operations successful")
