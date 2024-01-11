from sklearn.datasets import make_classification
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
#matplotlib.use('TkAgg')

def get_random_dataset(size: int) -> tuple[np.array, np.array]:
    '''
    Generate and return a random linearly separable dataset
    
    Args:
        size (int): the size of the dataset
    Returns:
        a tuple[data points, labels]
    '''
    isSeparable = False
    while not isSeparable:
        dataset = make_classification(n_samples=size, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
        dataset[1][dataset[1] == 0] = -1
        class1 = dataset[0][dataset[1] ==-1]
        class2 = dataset[0][dataset[1] == 1]
        isSeparable = any([class1[:, i].max() < class2[:, i].min() or class1[:, i].min() > class2[:, i].max() for i in range(2)])

    return dataset

def show_plot(points, w = None, show_db = False) -> None:
    '''
    Display the dataset `points`.
    If w is specified and show_db is True, then the decision boundary is also displayed
    '''
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        right=False,
        left=False,
        labelbottom=False,
        labeltop=False,
        labelright=False,
        labelleft=False
    )
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    class1 = points[0][points[1] == -1]
    class2 = points[0][points[1] == 1]

    plt.plot(class1[:, 0], class1[:, 1], marker='_', ls='', c='red')
    plt.plot(class2[:, 0], class2[:, 1], marker='+', ls='', c='green')

    if show_db == True:
        x_db = np.array([np.amin(points[0]), np.amax(points[0])])
        y_db_tmp = []
        for x in np.linspace(np.amin(points[0]),np.amax(points[0])):
            slope = 0 if w[2] == 0 else -w[1]/w[2]
            intercept = 0 if w[2] == 0 else -w[0]/w[2]

            y = (slope * x) + intercept
            y_db_tmp.append(y)

        y_db = np.array([min(y_db_tmp), max(y_db_tmp)])
        plt.plot(x_db, y_db, color="#1a1386")

    plt.show()
