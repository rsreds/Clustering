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
times = []
indeces = []
indexclust = []
for row in df.itertuples():
    cluster = getattr(row, 'cluster')
    time = getattr(row, 'time_cluster')
    index = getattr(row, 'Index') + 1
    times.append(time)
    indeces.append(index)
    if cluster > last_max:
        last_max = cluster
        indexclust.append(index)

timeset = [times[i-1] for i in indexclust]


plt.plot(times[:-1], indeces[:-1])
slope, intercept = np.polyfit(times, indeces, 1)

line = line = np.asarray(slope)*times+intercept
formula = "{:.0f}x + {:.0f}".format(slope, intercept)
#plt.plot(times, line, '--', alpha=0.8, label=formula)
plt.scatter(timeset, indexclust, marker='.')
plt.xlim(0, 150)
start, end = plt.xlim()
plt.ylim(0, 175000)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

plt.xlabel(r'time ($s$)')
plt.ylabel(r'loglines')

# plt.legend()

plt.show()
