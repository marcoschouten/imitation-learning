
# INPUT: INFERRED.CSV
# OUTPUT: ENCODED.CSV



import numpy as np
import matplotlib.pyplot as plt

import pandas as pd





# hyper parameters
n_steps = 50
threshold = 25
max_skip = 0.15* n_steps


# generate data
df = pd.read_csv('inferred.csv')
data_array = np.array(df.iloc[:,:])


T = np.linspace(-0, 1, n_steps)

# REAL DATASET
data = {}
data[0] = data_array[:,0]
data[1] = data_array[:,1]



# remove points according to gradient < threshold
N = np.shape(data)




def computeGrad(data, idx):
    t = data[0][:]
    dt = t[1] - t[0]
    y = data[1][:]
    grad = np.gradient(y, dt)
    laplacian = np.gradient(grad, dt)
    return np.abs(laplacian)




# FIND INDEXS TO SAVE________________
filteredIdx=[]
filteredIdx.append(0)

laplacian = computeGrad(data, 0)
counter_skipped = 0
for i in range(0, n_steps):
    if (laplacian[i] > threshold or counter_skipped > max_skip ):
        filteredIdx.append(i)
        counter_skipped = 0
    else:
        counter_skipped +=1

filteredIdx.append((n_steps-1))
print("len of reduced " + str(len(filteredIdx)))




#_________ PRINT DATA


x_f = data[0][filteredIdx]
y_f = data[1][filteredIdx]
t_f = T[filteredIdx]



plt.figure(figsize=(5, 5))
# plt.subplot(121)


plt.plot(data[0], data[1], c='blue',  label="inferred", lw=2)
plt.scatter(t_f, y_f, c='green',  label="encoded", lw=2)





df_reconstructed = pd.read_csv('re_traj.csv')
rec = []
rec.append( df_reconstructed.loc[:,'x1'])
rec.append( df_reconstructed.loc[:,'x2'])
plt.plot(rec[0], rec[1], c='orange', label="decoded" , lw=2)
plt.suptitle('', fontsize=20)
plt.xlabel('t', fontsize=10)
plt.ylabel('x1', fontsize=10)


filt_size = len(filteredIdx)

saved_percentage = (filt_size / n_steps )*100


print("datapoints saved " + str(saved_percentage))









plt.legend()
plt.show()

print(np.shape(data[0]))
print(np.shape(x_f))



with open('checkpoints.csv', 'w') as f:
    f.write('x1,x2\n')
    for i in range(0, filt_size):
        f.write('{},{}\n'.format(t_f[i], y_f[i]))






