
# INPUT:
# GROUND TRUTH
# INFERRED
# RECONSTRUCTED


from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interpolate


def align(y) :

    x = np.linspace(0, 1, 50)
    dt = x[1] - x[0]
    grad = np.gradient(y, dt)
    f = interpolate.CubicHermiteSpline(x, y, grad)
    xnew = np.linspace(0, 1, 70)
    ynew = f(xnew)  # use interpolation function returned by `interp1d`
    # plt.plot(x,y, xnew, ynew,)
    # plt.show()
    return ynew


#______________________________ MAIN ______________________________


n_dim = 2
df_ground_truth = pd.read_csv('ground_truth.csv')
df_inferred =  pd.read_csv('inferred.csv')
df_reconstructed = pd.read_csv('recovered_trajectory.csv')





# GT | RECONSTRUCTED       OVERALL
mae = np.zeros(n_dim)
mse = np.zeros(n_dim)
r2 = np.zeros(n_dim)
for i in range(0, n_dim):
    gt_i = align(df_ground_truth.iloc[:, i])
    in_i = align(df_inferred.iloc[:, i])
    re_i = align(df_reconstructed.iloc[:, i])
    mae[i] = metrics.mean_absolute_error(gt_i, re_i)
    mse[i] = metrics.mean_squared_error(gt_i, re_i)
    r2[i] = metrics.r2_score(gt_i,re_i)

with open('score_OVERALL.csv', 'w') as f:
    f.write('mae : {}'.format(mae))
    f.write('mse : {}'.format(mse))
    f.write('r2 : {}'.format(r2))



# GT | INFERRED       ML
mae = np.zeros(n_dim)
mse = np.zeros(n_dim)
r2 = np.zeros(n_dim)
for i in range(0, n_dim):
    gt_i = align(df_ground_truth.iloc[:, i])
    in_i = align(df_inferred.iloc[:, i])
    re_i = align(df_reconstructed.iloc[:, i])
    mae[i] = metrics.mean_absolute_error(gt_i, in_i)
    mse[i] = metrics.mean_squared_error(gt_i, in_i)
    r2[i] = metrics.r2_score(gt_i,in_i)

with open('score_ML.csv', 'w') as f:
    f.write('mae : {}'.format(mae))
    f.write('mse : {}'.format(mse))
    f.write('r2 : {}'.format(r2))



# INFERRED | RECONSTRUCTED     ENCODER
mae = np.zeros(n_dim)
mse = np.zeros(n_dim)
r2 = np.zeros(n_dim)
for i in range(0, n_dim):
    gt_i = align(df_ground_truth.iloc[:, i])
    in_i = align(df_inferred.iloc[:, i])
    re_i = align(df_reconstructed.iloc[:, i])
    mae[i] = metrics.mean_absolute_error(in_i, re_i)
    mse[i] = metrics.mean_squared_error(in_i, re_i)
    r2[i] = metrics.r2_score(in_i,re_i)

with open('score_ENCODER.csv', 'w') as f:
    f.write('mae : {}'.format(mae))
    f.write('mse : {}'.format(mse))
    f.write('r2 : {}'.format(r2))






