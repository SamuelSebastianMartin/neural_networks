#! /usr/bin/env python3


from matplotlib import pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = x**2

plt.plot(x, y)
plt.show()