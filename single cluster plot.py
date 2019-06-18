import os
import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt


path = 'C:/Users/simor/Google Drive/clustering/storm-fe/results-new/'
filename = 'clustered-storm-frontend-server.log-201812020.7.zip'

df = pd.read_csv(path + filename,
                 usecols=['timestamp', 'message',
                          'time_cluster', 'cluster', 'similarity'],
                 compression='zip')
last_max = 0
x = []
y = []
for row in df.itertuples():
    cluster = getattr(row, 'cluster')
    if cluster > last_max:
        last_max = cluster
        index = getattr(row, 'Index') + 1
        x.append(index)
        y.append(last_max)
x.append(df.last_valid_index())
y.append(last_max)


plt.plot(x, y)
plt.scatter(x[:-1], y[:-1], s=30, marker='.')

yint = range(min(y), ceil(max(y))+1)
plt.yticks(yint)
plt.ylabel('clusters')

plt.xticks(np.arange(0, max(x), 250000))
plt.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
plt.xlabel('log lines')


plt.show()
