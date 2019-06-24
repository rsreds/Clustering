import numpy as np
from cluster_creator import similarity

strs = [("Test sample",                   "Test values"),
        ("Test sample",                   "Test smaple"),
        ("Test sample test",  "Test sample")]

for t in strs:
    sims = [str(similarity(t[0], t[1], met))
            for met in ['jaccard', 'cosine', 'levenshtein']]
    print(t[0]+'\n'+t[1]+' \t ' + sims[0] + ' \t ' +
          sims[1] + ' \t ' + sims[2] + ' \n')
