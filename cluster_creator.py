import regex as re
import numpy as np
import pandas as pd
from time import time
from math import cos, sqrt
from tqdm import tqdm
from Levenshtein import seqratio
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MIN_SIM_THRESHOLD = 0.7

USER = re.compile('<.*>')
REQUEST = re.compile(
    r'((\w|\d){8})-((\w|\d){4})-((\w|\d){4})-((\w|\d){4})-((\w|\d){12})')
TOKEN = re.compile(r'\[token:.*\]')
SURL = re.compile(r'srm:.+?(?=]| |$)')
PATH = re.compile(r'/.+?(?=]| |$)')
IP = re.compile(
    r'(ffff:(\d){1,3}.(\d){1,3}.(\d){1,3}.(\d){1,3})|(((\S){1,4}:){3,4}:(\S){1,4}:(\S){1,4})|(((\S){1,4}:){4,}(\S){1,4})')
MAIL = re.compile(r'(\S)+\@\S+\.\S+')


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
        leven = seqratio(str1.lower(), str2.lower())
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
