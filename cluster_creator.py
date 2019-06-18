import sys
import getopt
from time import time
from math import cos, sqrt
import regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import Levenshtein as lv
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MIN_SIM_THRESHOLD = 0.7

USER = re.compile('<.*>')
REQUEST = re.compile(
    '((\w|\d){8})-((\w|\d){4})-((\w|\d){4})-((\w|\d){4})-((\w|\d){12})')
TOKEN = re.compile('\[token:.*\]')
SURL = re.compile('srm:.+?(?=]| |$)')
PATH = re.compile('/.+?(?=]| |$)')
IP = re.compile(
    '(ffff:(\d){1,3}.(\d){1,3}.(\d){1,3}.(\d){1,3})|(((\S){1,4}:){3,4}:(\S){1,4}:(\S){1,4})|(((\S){1,4}:){4,}(\S){1,4})')
MAIL = re.compile('(\S)+\@\S+\.\S+')


def clean_log(log):
    '''
    Cleans the log file using regex to remove unneccessary data
    '''
    log = re.sub(USER, 'USER', log)
    log = re.sub(TOKEN, 'TOKEN', log)
    log = re.sub(REQUEST, 'REQ_ID', log)
    log = re.sub(SURL, 'SURL', log)
    log = re.sub(PATH, 'PATH', log)
    log = re.sub(IP, 'IP', log)
    log = re.sub(MAIL, 'MAIL', log)
    return log


def get_vectors(str1, str2):
    '''
    Transofrm strings into an array of occurrences for each word
    '''
    text = [str1, str2]
    vectorizer = CountVectorizer()
    result = vectorizer.fit_transform(text)
    return result.toarray()


def similarity(str1, str2, method='levenshtein'):
    '''
    Similarity functions
    '''
    if method == 'levenshtein':
        leven = lv.seqratio(str1.lower(), str2.lower())
        return leven
    if method == 'jaccard':
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    if method == 'cosine':
        vectors = get_vectors(str1, str2)
        cosine = cosine_similarity(vectors)
        return cosine[0, 1]


def clusterize(df, sim_thres=DEFAULT_MIN_SIM_THRESHOLD):
    '''
    Clusterizes the dataframe based on the similarity of logs to the reference 
    of each cluster
    If no cluster within the threshold is found creates new cluster

    Returns the dataframe and a dicionary of clusters
    '''
    t0 = time()
    clusters = []
    id = 0
    for row in tqdm(df.itertuples()):
        best_clust = None
        i = getattr(row, 'Index')
        datetime = getattr(row, 'datetime')
        log = getattr(row, 'message')
        cleaned_log = clean_log(log)
        t = time()-t0
        df.at[i, 'time_cluster'] = t
        if pd.isnull(datetime):
            cleaned_log = 'JAVA_ERROR'
        if len(clusters) == 0:
            clusters.append({'id': id, 'ref': cleaned_log, 'count': 1})
            df.at[i, 'cluster'] = id
            df.at[i, 'similarity'] = 1
            id = id+1
            continue

        similarities = [similarity(cleaned_log, cluster['ref'], 'levenshtein')
                        for cluster in clusters]
        best_clust = np.argmax(similarities)
        if similarities[best_clust] > sim_thres:
            clusters[best_clust]['count'] = clusters[best_clust]['count']+1
            df.at[i, 'cluster'] = clusters[best_clust]['id']
            df.at[i, 'similarity'] = similarities[best_clust]
        else:
            clusters.append({'id': id, 'ref': cleaned_log, 'count': 1})
            df.at[i, 'cluster'] = id
            df.at[i, 'similarity'] = 1
            id = id+1

    df.cluster = df.cluster.astype(int)
    return df, clusters


def add_value_labels(ax, reference_logs, spacing=5):
    '''
    Writes the occurrence value over each bar

    If reference log list is provided writes the corresponding log over each bar
    '''
    id = 0
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        vert_spacing = spacing
        vert_alignment = 'bottom'
        angle = 90

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            vert_spacing *= -1
            vert_alignment = 'top'

        # Create value annotation
        label = "{:.0f}".format(y_value)
        ax.annotate(label, (x_value, y_value), xytext=(0, vert_spacing),
                    textcoords="offset points", ha='center', va=vert_alignment)

        # Create log annotation
        if isinstance(reference_logs, (list,)):
            label = reference_logs[id]
            ax.annotate(label, (x_value, y_value), xytext=(0, vert_spacing*4),
                        textcoords="offset points", ha='center', va=vert_alignment,
                        rotation=angle, fontsize='xx-small')
        id = id+1


def plot_clusters(cluster_array, write_ref=False, skip_single=False):
    '''
    Plot the cluster size bar graph
    If list of label is passed prints it over each bar

    Label should be the refernce log for each cluster
    '''
    fig, ax = plt.subplots()
    ids = [row[0] for row in cluster_array]
    occurrence = [row[1] for row in cluster_array]

    if write_ref is True:
        reference_log = [row[2] for row in cluster_array]
    else:
        reference_log = None

    if skip_single == True:
        y = occurrence[occurrence > 1]
        x = ids[occurrence > 1]
    else:
        y = occurrence
        x = ids

    ax.bar(range(len(x)), y)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    ax.set_xlabel('cluster')
    ax.set_ylabel('occurrences')

    add_value_labels(ax, reference_log)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

    return fig


def run():
    '''
    Read csv from inputfile as dataframe
    Clusterizes the csv based on the give minimum similrity threshold
    Save the resulting .csv on the outputfile, with cluster column indicating the corrisponding cluster for each log
    Plots the cluster size bar graoh
    '''
    inputpath = 'C:\\Users\\simor\\Google Drive\\clustering\\storm-fe\\'
    outputpath = 'C:\\Users\\simor\\Google Drive\\clustering\\storm-fe\\results-new\\'

    thresholds = [0.40, 0.50, 0.60, 0.625, 0.65, 0.675, 0.70, 0.80, 0.90]
    filelist = ['storm-frontend-server.log-20181202.csv.zip',
                'storm-frontend-server.log-20181207.csv.zip', ]
    for filename in filelist:
        for threshold in thresholds:
            print('Loading: ' + filename)
            inputfile = inputpath + filename
            savename = filename[:-8] + str(threshold)
            df = pd.read_csv(inputfile, compression='zip')
            print('Loaded ' + str(len(df.index)) + ' lines')

            df, cluster_dict = clusterize(
                df, threshold)
            print('Clustered. Saving to ' + outputpath)

            array = [[dic['id'], dic['count'], dic['ref']]
                     for dic in cluster_dict]
            np.savetxt(outputpath + 'cluster_table-' + savename + '.csv',
                       array, fmt='%s', delimiter=',')
            print('Saved clusters csv in ' +
                  'cluster_table-' + savename + '.csv')

            df.to_csv(outputpath + 'clustered-' +
                      savename + '.zip', compression='zip')

            print('Saved log csv in ' + 'clustered-' + savename + '.zip')


def main(argv):
    run()


if __name__ == "__main__":
    main(sys.argv[1:])
