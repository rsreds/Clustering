import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def add_value_labels(ax, write_occ, reference_logs):
    '''
    Writes the occurrence value over each bar

    If reference log list is provided writes the corresponding log over each bar
    '''
    id = 0
    if write_occ:
        vert_spacing = 5
    else:
        vert_spacing = 0
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        vert_alignment = 'bottom'
        angle = 90

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            vert_spacing *= -1
            vert_alignment = 'top'
        if write_occ:
            # Create value annotation
            label = "{:.0f}".format(y_value)
            ax.annotate(label, (x_value, y_value), xytext=(0, vert_spacing),
                        textcoords="offset points", ha='center', va=vert_alignment)

        # Create log annotation
        if isinstance(reference_logs, (list,)):
            label = reference_logs[id]
            ax.annotate(label, (x_value, y_value), xytext=(0, 5 + vert_spacing*3),
                        textcoords="offset points", ha='center', va=vert_alignment,
                        rotation=angle, fontsize='xx-small')
        id = id+1


def plot_clusters(cluster_array, write_occ=False,  write_ref=False, skip_single=False, ordered=True):
    '''
    Plot the cluster size bar graph
    If list of label is passed prints it over each bar

    Label should be the refernce log for each cluster
    '''
    fig, ax = plt.subplots()
    if ordered:
        cluster_array[::-1].sort(order=['f1'], axis=0)
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

    add_value_labels(ax, write_occ, reference_log)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

    return fig
