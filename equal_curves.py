import time
import numpy as np
import pandas as pd
from numpy import savetxt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# Get the fitted data.
x_raw = pd.read_csv('x_regfit.csv')
x_raw = x_raw.drop(x_raw.columns[[0]], axis=1)
#x_raw = x_raw.drop(x_raw.index[0])

y_raw = pd.read_csv('y_regfit.csv')
y_raw = y_raw.drop(y_raw.columns[[0]], axis=1) 
#y_raw = y_raw.drop(y_raw.index[0])

# Calculate length of reference frame and the length of each segment.
x0 = x_raw.iloc[0,:]
y0 = y_raw.iloc[0,:]

L = 0
for i in range(0, (len(x0)-1), 1):
    L = L + math.dist((x0[i], y0[i]), (x0[i+1], y0[i+1]))

n = 30
dL = L/n

X_scaled = []
Y_scaled = []

for i in range (1, len(x_raw), 1):
    x = x_raw.iloc[i,:]
    y = y_raw.iloc[i,:]

    X_fit = [None]*n
    Y_fit = [None]*n

    X_fit[0] = x[0]
    Y_fit[0] = y[0]

    X_fit[-1] = x[-1]
    Y_fit[-1] = y[-1]

    L0 = 0
    k = 1

    for j in range(0, (len(x)-1), 1):
        L0 = L0 + math.dist((x[j], y[j]), (x[j+1], y[j+1]))
        if (L0>=(k*dL)):
            X_fit.append(x[j])
            Y_fit.append(y[j])
            k = k + 1
    X_scaled.append(X_fit)
    Y_scaled.append(Y_fit)

fig, axs = plt.subplots()
axs.set_xlim((0, 3000))
axs.set_ylim((0, 1500))
   
beam, = axs.plot(X_scaled[0], Y_scaled[0], 'o')

def animate(i):
    targetx = np.array(X_scaled[i])
    targety = np.array(Y_scaled[i])
    beam.set_data( [targetx, targety])
    return beam,
scaled_animation = FuncAnimation(fig, animate, frames=len(X_scaled), blit=True)
plt.show()

scaled_animation.save('7.2Hz_400mV2.gif', writer='pillow', fps=15)