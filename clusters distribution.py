import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor

path = 'C:/Users/simor/Google Drive/clustering/storm-fe/results-new/'

labels = [0.4, 0.5, 0.6, 0.625, 0.65, 0.675, 0.7, 0.8, 0.9]
filelist = ['clustered-storm-frontend-server.log-201812070.4.zip', 'clustered-storm-frontend-server.log-201812070.5.zip',
            'clustered-storm-frontend-server.log-201812070.6.zip', 'clustered-storm-frontend-server.log-201812070.625.zip',
            'clustered-storm-frontend-server.log-201812070.65.zip', 'clustered-storm-frontend-server.log-201812070.675.zip',
            'clustered-storm-frontend-server.log-201812070.7.zip', 'clustered-storm-frontend-server.log-201812070.8.zip',
            'clustered-storm-frontend-server.log-201812070.9.zip']
clusters = []
dfs = []
for file in os.listdir(path):
    if file in filelist:
        df = pd.read_csv(path + file,
                         usecols=['timestamp', 'message', 'time_cluster', 'cluster', 'similarity'], compression='zip')
        dfs.append(df)
        clusters.append(df['cluster'].max() + 1)
    else:
        continue


""" plt.figure()
for i, df in enumerate(dfs):
    plt.plot(df['time_cluster'], df.index)
    slope, intercept = np.polyfit(df['time_cluster'], df.index, 1)
    print('{:.2f} & {:.0f} & {:.0f}\\\\'.format(labels[i], clusters[i], slope))
    plt.legend(labels, title="Similarity Threshold")
plt.xlabel(r'time ($s$)')
plt.ylabel(r'log lines')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
plt.show() """


variances = []
means = []
p90 = []
p95 = []
p80 = []
for df in dfs:
    q = 1
    percentile = floor(q * (len(df.index))) - 1
    ordered_clusters = (df['cluster'].value_counts()).index.values
    df['cluster'] = pd.Categorical(df.cluster, categories=ordered_clusters)
    df = df.sort_values('cluster')
    stopclust = df['cluster'].iat[percentile]
    stopindex = np.where(ordered_clusters == stopclust)[0][0] + 1
    var = [np.var(df[df['cluster'] == cluster]['similarity'])
           for cluster in ordered_clusters[:stopindex]]
    variances.append(var)
    means.append(np.mean(var))

    q = 0.95
    percentile = floor(q * (len(df.index))) - 1
    ordered_clusters = (df['cluster'].value_counts()).index.values
    df['cluster'] = pd.Categorical(df.cluster, categories=ordered_clusters)
    df = df.sort_values('cluster')
    stopclust = df['cluster'].iat[percentile]
    stopindex = np.where(ordered_clusters == stopclust)[0][0] + 1
    var = [np.var(df[df['cluster'] == cluster]['similarity'])
           for cluster in ordered_clusters[:stopindex]]
    variances.append(var)
    p95.append(np.mean(var))

    q = 0.90
    percentile = floor(q * (len(df.index)))-1
    ordered_clusters = (df['cluster'].value_counts()).index.values
    df['cluster'] = pd.Categorical(df.cluster, categories=ordered_clusters)
    df = df.sort_values('cluster')
    stopclust = df['cluster'].iat[percentile]
    stopindex = np.where(ordered_clusters == stopclust)[0][0] + 1
    var = [np.var(df[df['cluster'] == cluster]['similarity'])
           for cluster in ordered_clusters[:stopindex]]
    variances.append(var)
    p90.append(np.mean(var))

    q = 0.80
    percentile = floor(q * (len(df.index)))-1
    ordered_clusters = (df['cluster'].value_counts()).index.values
    df['cluster'] = pd.Categorical(df.cluster, categories=ordered_clusters)
    df = df.sort_values('cluster')
    stopclust = df['cluster'].iat[percentile]
    stopindex = np.where(ordered_clusters == stopclust)[0][0] + 1
    var = [np.var(df[df['cluster'] == cluster]['similarity'])
           for cluster in ordered_clusters[:stopindex]]
    variances.append(var)
    p80.append(np.mean(var))

plt.figure()
ax1 = plt.subplot()
ax1.plot(labels, means, '-o', label='p100', alpha=0.8, ms=3)
ax1.plot(labels, p95, '-o', label='p95', alpha=0.8, ms=3)
ax1.plot(labels, p90, '-o', label='p90', alpha=0.8, ms=3)
ax1.plot(labels, p80, '-o', label='p80', alpha=0.8, ms=3)


ax1.set_xticks(labels)
ax1.set_xticklabels(labels, rotation='vertical')
ax1.set_xlabel('similarity threshold')
ax1.set_ylabel('mean cluster variance')
plt.legend()

ax2 = ax1.twiny()
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(clusters)
ax2.spines['bottom'].set_position(('outward', 50))
ax2.set_xlabel('clusters')
ax2.set_xlim(ax1.get_xlim())

plt.show()
