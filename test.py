import numpy as np

a = np.array([1, 1, 2])
b = np.array([1, 1, 3])

print((a == b).all())
print((a == b).any())