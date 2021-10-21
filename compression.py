
import numpy as np
import matplotlib.pyplot as plt


# generate data
n_steps = 50
sigma=0.25
mu=0.5
T = np.linspace(-0, 1, n_steps)
data = (T / 20 + 1 / (sigma * np.sqrt(2 * np.pi)) *  np.exp(-0.5 * ((T - mu) / sigma) ** 2))

plt.plot(T,data )
plt.show()

