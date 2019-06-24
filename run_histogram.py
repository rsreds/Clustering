import pandas as pd
import os
import numpy as np
from cluster_histogram import plot_clusters


def run(inputpath):
    filelist = ['cluster_table-storm-frontend-server.log-201812020.7.csv',
                'cluster_table-storm-frontend-server.log-20181202-sample0.csv',
                'cluster_table-storm-frontend-server.log-20181202-sample1.csv',
                'cluster_table-storm-frontend-server.log-20181202-sample2.csv']

    filelist = ['cluster_table-storm-frontend-server.log-201812070.675.csv']

    for filename in filelist:
        inputfile = inputpath + filename
        array = np.loadtxt(inputfile, dtype=("i, i, U100"), delimiter=',')
        print(array)
        plot_clusters(array, write_occ=False, write_ref=True)


if __name__ == "__main__":
    inputpath = 'C:\\Users\\simor\\Google Drive\\clustering\\output\\storm-fe\\20181207\\'
    run(inputpath)
