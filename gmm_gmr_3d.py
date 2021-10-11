import pickle
import matplotlib.pyplot as plt
from gmm_gmr.mixtures import GMM_GMR
import numpy as np

############### function 1
def make_demonstrations(n_demonstrations, n_steps, sigma=0.25, mu=0.5,
                        start=np.zeros(3), goal=np.ones(3), random_state=None, n_task_dims=3):
    """Generates demonstration that can be used to test imitation learning.
    Parameters
    ----------
    n_demonstrations : int
        Number of noisy demonstrations
    n_steps : int
        Number of time steps
    sigma : float, optional (default: 0.25)
        Standard deviation of noisy component
    mu : float, optional (default: 0.5)
        Mean of noisy component
    start : array, shape (2,), optional (default: 0s)
        Initial position
    goal : array, shape (2,), optional (default: 1s)
        Final position
    random_state : int
        Seed for random number generator
    Returns
    -------
    X : array, shape (n_task_dims, n_steps, n_demonstrations)
        Noisy demonstrated trajectories
    ground_truth : array, shape (n_task_dims, n_steps)
        Original trajectory
    """
    random_state = np.random.RandomState(random_state)

    X = np.empty((n_task_dims, n_steps, n_demonstrations))

    # Generate ground-truth for plotting
    ground_truth = np.empty((n_task_dims, n_steps))
    T = np.linspace(-0, 1, n_steps)
    ground_truth[0] = (T / 20 + 1 / (sigma * np.sqrt(2 * np.pi)) *
                       np.exp(-0.5 * ((T - mu) / sigma) ** 2))
    ground_truth[1] = (T / 20 + 1 / (sigma * np.sqrt(2 * np.pi)) *
                       np.exp(-0.5 * ((T - mu) / sigma) ** 2))
    ground_truth[2] = (T / 20 + 1 / (sigma * np.sqrt(2 * np.pi)) *
                       np.exp(-0.5 * ((T - mu) / sigma) ** 2))

    # Generate trajectories
    for i in range(n_demonstrations):
        noisy_sigma = sigma * random_state.normal(1.0, 0.1)
        noisy_mu = mu * random_state.normal(1.0, 0.1)
        X[0, :, i] = T + (1 / (noisy_sigma * np.sqrt(2 * np.pi)) *
                          np.exp(-0.5 * ((T - noisy_mu) /
                                         noisy_sigma) ** 2))
        X[1, :, i] = T + (1 / (noisy_sigma * np.sqrt(2 * np.pi)) *
                          np.exp(-0.5 * ((T - noisy_mu) /
                                         noisy_sigma) ** 2))
        X[2, :, i] = T + (1 / (noisy_sigma * np.sqrt(2 * np.pi)) *
                          np.exp(-0.5 * ((T - noisy_mu) /
                                         noisy_sigma) ** 2))

    # Spatial alignment
    current_start = ground_truth[:, 0]
    current_goal = ground_truth[:, -1]
    current_amplitude = current_goal - current_start
    amplitude = goal - start
    ground_truth = ((ground_truth.T - current_start) * amplitude /
                    current_amplitude + start).T

    for demo_idx in range(n_demonstrations):
        current_start = X[:, 0, demo_idx]
        current_goal = X[:, -1, demo_idx]
        current_amplitude = current_goal - current_start
        X[:, :, demo_idx] = ((X[:, :, demo_idx].T - current_start) *
                             amplitude / current_amplitude + start).T

    return X, ground_truth






#_____________________________________________ get data
# opt a
ee_poses = pickle.load(open('data/example_data.pk', 'rb'))




# opt b
X, _ = make_demonstrations(n_demonstrations=10, n_steps=50, random_state=0, n_task_dims=3)
X = X.transpose(2, 1, 0)



# choose data
data = ee_poses


# N = data.shape[1]
data_shape = np.shape(data[0])





#_____________________________________________ build model
latent_space_dim = 3
gmm_gmr = GMM_GMR(data, latent_space_dim)# param: trajectory, dimensionality of output trajectory
gmm_gmr.fit()












#_____________________________________________ plots
f, axarr = plt.subplots(3, 1)

for i in range(len(data)):
    for j in range(latent_space_dim):
        axarr[j].plot(gmm_gmr.trajectories[i,:,j], linestyle=':')

for j in range(latent_space_dim):
    axarr[j].scatter(gmm_gmr.centers_temporal, gmm_gmr.centers_spatial[:,j], label='centers')

times, trj = gmm_gmr.generate_trajectory(0.1)
for j in range(latent_space_dim):
    axarr[j].plot(times, trj[:,j], label='estimated',c='r')



plt.legend()
plt.show()