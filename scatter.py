import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

df = pd.read_csv(
    'C:\\Users\\simor\\Google Drive\\clustering\\output\\clustered-2019-05-24-storm-backend.log.csv', index_col=0)

plt.scatter(df.index, df.cluster, s=5, marker='.')
plt.show()
