import numpy as np
import math
import time
import random

N = 200000
X = [np.random.random() for _ in range(N)]

START = time.time()
for x in X:
    # np.deg2rad(x)
    np.square(x)
print("Numpy", time.time() - START)

START = time.time()
for x in X:
    math.pow(x, 2)
    # math.radians(x)

print("Math", time.time() - START)

START = time.time()
for x in X:
    # math.pow(x, 2)\
    # math.radians(x)
    x * x
print("Math2", time.time() - START)