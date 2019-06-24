import sys
import pandas as pd
import numpy as np
from cluster_creator import clusterize


def run():
    '''
    Read csv from inputfile as dataframe
    Clusterizes the csv based on the give minimum similrity threshold
    Save the resulting .csv on the outputfile, with cluster column indicating the corrisponding cluster for each log
    Plots the cluster size bar graoh
    '''
    inputpath = 'C:\\Users\\simor\\Google Drive\\clustering\\storm-fe\\'
    outputpath = 'C:\\Users\\simor\\Google Drive\\clustering\\output\\measures\\'

    thresholds = [0.5, 0.6, 0.7, 0.8]
    measures = ['jaccard', 'levenshtein', 'cosine']
    filelist = ['storm-frontend-server.log-20181202.csv.zip']
    for filename in filelist:
        for threshold in thresholds:
            for measure in measures:
                # Load csv
                print('Loading: ' + filename)
                inputfile = inputpath + filename
                savename = measure + '-' + filename[:-8] + '-' + str(threshold)
                df = pd.read_csv(
                    inputfile, compression='zip').sample(frac=0.01, random_state=1)
                print('Loaded ' + str(len(df.index)) + ' lines')

                # run clusterization
                print('Clustering with threshold: ' +
                      str(threshold) + ' and measure: ' + measure)
                df, cluster_dict = clusterize(
                    df, threshold, measure)
                print('Clustered. Saving to ' + outputpath)

                # Save cluster table with occurrences and ref
                array = [[dic['id'], dic['count'], dic['ref']]
                         for dic in cluster_dict]
                np.savetxt(outputpath + 'cluster_table-' + savename + '.csv',
                           array, fmt='%s', delimiter=',')
                print('Saved clusters csv in ' +
                      'cluster_table-' + savename + '.csv')

                # Save clustered df
                df.to_csv(outputpath + 'clustered-' +
                          savename + '.zip', compression='zip')

                print('Saved log csv in ' + 'clustered-' + savename + '.zip')


def main(argv):
    run()


if __name__ == "__main__":
    main(sys.argv[1:])
