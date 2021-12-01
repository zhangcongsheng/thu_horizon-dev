import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, datetime

x = np.array([1, 2, 3, 4, 5])
y = x * x

plt.plot(x, y)

text_data = [["1", "1"], ["1", "1"]]

plt.table(cellText=text_data, colWidths=[0.1] * 3, loc="best")

plt.show()
