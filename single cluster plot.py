import os
import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt


W = 6.5
H = W / 1.618
FIGSIZE = (W, H)

path = 'C:/Users/simor/Google Drive/clustering/output/storm-fe/20181207/'
filename = 'clustered-storm-frontend-server.log-201812070.675.zip'

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

plt.figure(figsize=FIGSIZE)
plt.plot(x, y)
plt.scatter(x[:-1], y[:-1], s=30, marker='.')

yint = range(min(y), ceil(max(y))+1)
plt.yticks(yint)
plt.ylabel('clusters')

plt.xticks(np.arange(0, max(x), 250000))
plt.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
plt.xlabel('log lines')
plt.tight_layout()

plt.show()
