"""
===============================
Learn Time-Indexed Trajectories
===============================
We learn a GMM from multiple similar trajectories that consist of points
(t, x_1, x_2), where t is a time variable and x_1 and x_2 are 2D coordinates.
The GMM is initialized from a Bayesian GMM of sklearn to get a better fit
of the data, which is otherwise difficult in this case, where we have discrete
steps in the time dimension and x_1.
We compare the 95 % confidence interval in x_2 between the original data and
the learned GMM.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import BayesianGaussianMixture
from itertools import cycle
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
from gmr.utils import check_random_state


# def make_demonstrations(n_demonstrations, n_steps, sigma=0.25, mu=0.5,
#                         start=np.zeros(2), goal=np.ones(2), random_state=None):
#     """Generates demonstration that can be used to test imitation learning.
#     Parameters
#     ----------
#     n_demonstrations : int
#         Number of noisy demonstrations
#     n_steps : int
#         Number of time steps
#     sigma : float, optional (default: 0.25)
#         Standard deviation of noisy component
#     mu : float, optional (default: 0.5)
#         Mean of noisy component
#     start : array, shape (2,), optional (default: 0s)
#         Initial position
#     goal : array, shape (2,), optional (default: 1s)
#         Final position
#     random_state : int
#         Seed for random number generator
#     Returns
#     -------
#     X : array, shape (n_task_dims, n_steps, n_demonstrations)
#         Noisy demonstrated trajectories
#     ground_truth : array, shape (n_task_dims, n_steps)
#         Original trajectory
#     """
#     random_state = np.random.RandomState(random_state)
#
#     X = np.empty((2, n_steps, n_demonstrations))
#
#     # Generate ground-truth for plotting
#     ground_truth = np.empty((2, n_steps))
#     T = np.linspace(-0, 1, n_steps)
#     ground_truth[0] = T
#     ground_truth[1] = (T / 20 + 1 / (sigma * np.sqrt(2 * np.pi)) *
#                        np.exp(-0.5 * ((T - mu) / sigma) ** 2))
#
#     # Generate trajectories
#     for i in range(n_demonstrations):
#         noisy_sigma = sigma * random_state.normal(1.0, 0.1)
#         noisy_mu = mu * random_state.normal(1.0, 0.1)
#         X[0, :, i] = T
#         X[1, :, i] = T + (1 / (noisy_sigma * np.sqrt(2 * np.pi)) *
#                           np.exp(-0.5 * ((T - noisy_mu) /
#                                          noisy_sigma) ** 2))
#
#     # Spatial alignment
#     current_start = ground_truth[:, 0]
#     current_goal = ground_truth[:, -1]
#     current_amplitude = current_goal - current_start
#     amplitude = goal - start
#     ground_truth = ((ground_truth.T - current_start) * amplitude /
#                     current_amplitude + start).T
#
#     for demo_idx in range(n_demonstrations):
#         current_start = X[:, 0, demo_idx]
#         current_goal = X[:, -1, demo_idx]
#         current_amplitude = current_goal - current_start
#         X[:, :, demo_idx] = ((X[:, :, demo_idx].T - current_start) *
#                              amplitude / current_amplitude + start).T
#
#     return X, ground_truth



######################## GENERATE DATASET
def make_demonstrations3d(n_demonstrations, n_steps, sigma=0.25, mu=0.5,
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
                                         noisy_sigma) ** 2)) + np.random.random_sample()
        X[1, :, i] = T + (1 / (noisy_sigma * np.sqrt(2 * np.pi)) *
                          np.exp(-0.5 * ((T - noisy_mu) /
                                         noisy_sigma) ** 2)) + np.random.random_sample()
        X[2, :, i] = T + (1 / (noisy_sigma * np.sqrt(2 * np.pi)) *
                          np.exp(-0.5 * ((T - noisy_mu) /
                                         noisy_sigma) ** 2)) + np.random.random_sample()

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


plot_covariances = True
X, _ = make_demonstrations3d(
    n_demonstrations=10, n_steps=50,
    random_state=0)
X = X.transpose(2, 1, 0)
steps = X[:, :, 0].mean(axis=0)
expected_mean = X[:, :, 1].mean(axis=0)
expected_std = X[:, :, 1].std(axis=0)


# add time to the dataset
n_demonstrations, n_steps, n_task_dims = X.shape
X_train = np.empty((n_demonstrations, n_steps, n_task_dims + 1))
X_train[:, :, 1:] = X
t = np.linspace(0, 1, n_steps)
X_train[:, :, 0] = t


# stack all demos on top of each other.
X_train = X_train.reshape(n_demonstrations * n_steps, n_task_dims + 1)


# initalize model parameter
random_state = check_random_state(0)
n_components = 5





initial_means = kmeansplusplus_initialization(X_train, n_components, random_state)
initial_covs = covariance_initialization(X_train, n_components)
bgmm = BayesianGaussianMixture(n_components=n_components, max_iter=100).fit(X_train)
gmm = GMM(
    n_components=n_components,
    priors=bgmm.weights_,
    means=bgmm.means_,
    covariances=bgmm.covariances_,
    random_state=random_state)







####################### PLOT DATASET ###################################
f, axarr = plt.subplots(3, 1)
for i in range(n_demonstrations):
    for j in range(n_task_dims):
        axarr[j].plot(t, X[i,:,j], linestyle=':')


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[1,:,0], X[1,:,1], X[1,:,2], linestyle='-' ) # plot the point (2,3,4) on the figure
# plt.show()




means_over_time = []
y_stds = []
for step in t:
    conditional_gmm = gmm.condition([0], np.array([step]))
    conditional_mvn = conditional_gmm.to_mvn()
    means_over_time.append(conditional_mvn.mean)
    y_stds.append(np.sqrt(conditional_mvn.covariance[1, 1]))
    samples = conditional_gmm.sample(100)
    #plt.scatter(samples[:, 0], samples[:, 1], s=1)
    # for j in range(n_task_dims):
    #     axarr[j].plot(t, samples[:, j], linestyle='none', c="b")
means_over_time = np.array(means_over_time)
y_stds = np.array(y_stds)





# PLOT MEAN
for j in range(n_task_dims):
    axarr[j].plot(t, means_over_time[:, j], linestyle='-', c="r", lw=2)




plt.show()