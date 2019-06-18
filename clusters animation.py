import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

df = pd.read_csv('clustered-2019-06-02-storm-backend.log.csv', index_col=0)

size = df['cluster'].max()
matrix = []
data = np.zeros(size+1)
i = 0
for row in df.itertuples():
    i = i+1
    clus = getattr(row, 'cluster')
    data[clus] = data[clus] + 1
    if i == 1000:
        matrix.append(list(data))
        i = 0

fig = plt.figure()
barcollection = plt.bar(range(size+1), matrix[-1])


def animate(i):
    global matrix
    y = matrix[i]
    for i, b in enumerate(barcollection):
        b.set_height(y[i])


anim = animation.FuncAnimation(fig, animate, repeat=False, blit=False, frames=len(matrix),
                               interval=1)
anim.save('mymovie.mp4', writer=animation.FFMpegWriter(fps=10))
plt.show()
