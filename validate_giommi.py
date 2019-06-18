import os
import pandas as pd
import numpy as np
from math import ceil
import regex as re
import matplotlib.pyplot as plt


path = 'C:/Users/simor/Google Drive/clustering/storm-fe/results/'
filename = 'clustered-storm-frontend-server.log-201812020.68.zip'

df = pd.read_csv(path + filename,
                 usecols=['timestamp', 'message',
                          'time_cluster', 'cluster', 'similarity'],
                 compression='zip')


number = np.zeros(len(df['cluster'].unique()))
for row in df.itertuples():
    cluster = getattr(row, 'cluster')
    message = getattr(row, 'message')
    request = re.search(
        r'((?<=request \')|(?<=Request \')|(?<=REQUEST \')|(?<=Request: )|(?<=process_request : )).*?(?=\'|\.| from)', message, re.M | re.I)
    if request:
        request = request.group()
        if request.lower() == 'abort request':
            number[cluster] = number[cluster] + 1
print(number)
