"""
Plotting functions
"""

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def distplot(data, x):
    """
    Recreates the (deprecated) sns.distplot function
    """
    sns.histplot(data=data, x=x, kde=True, stat='density', linewidth=0)


def confusionplot(y_true, y_pred, labels):
    
    confusion = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(
        pd.DataFrame(confusion), annot=True, cmap='YlGnBu', fmt='g'
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(np.arange(len(labels)) + 0.5, labels)
    plt.yticks(np.arange(len(labels)) + 0.5, labels)

