

import  numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


T = np.linspace(0, 1, 50)
y=  -854112.8 + (4.389788 - -854112.8) / (1 + (T / 31.3205) ** 6.971958)
y = scale(y, axis=0, with_mean=True, with_std=True, copy=True )

random_state = np.random.RandomState(None)
sigma = 0.25
mu = 0.5
noisy_sigma = sigma * random_state.normal(1.0, 0.1)
noisy_mu = mu * random_state.normal(1.0, 0.1)
noiseA = T + (1 / (noisy_sigma * np.sqrt(2 * np.pi)) *
     np.exp(-0.5 * ((T - noisy_mu) /
                    noisy_sigma) ** 2))



noise_Scale = scale(noiseA, axis=0, with_mean=True, with_std=True, copy=True )


demo_s = y + 0.1 * noiseA


plt.plot(T, y, c="b", alpha=0.1)
plt.plot(T, demo_s, c="r", alpha=0.3)
plt.show()