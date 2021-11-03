


# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


fig = plt.figure()
ax = plt.axes(projection='3d')

# plot kinect
df = pd.read_csv('re_traj.csv')
x1 = np.array(df.loc[:, 'x2'])
x2 = np.array(df.loc[:, 'x2'])
t = np.linspace(0,1,50)
ax.plot3D(x1 ,x2, t, 'g', label='kinect')

plt.show()

#
#
# # plot inferred
# df = pd.read_csv('inferred.csv')
# x1 = np.array(df.iloc[:, 'x1'])
# x2 = np.array(df.iloc[:, 'x2'])
# t = np.linspace(0,1,50)
# ax.plot3D(x1, x2, t, 'r', label='kinect')
#
#
# # plot reconstructed
# df = pd.read_csv('recovered_trajectory.csv')
# x1 = np.array(df.iloc[:, 'x1'])
# x2 = np.array(df.iloc[:, 'x2'])
# t = np.linspace(0,1,50)
# ax.plot3D(x1, x2, t, 'm', label='kinect')
#
#
#
# # Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

