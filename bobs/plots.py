"""
Plotting functions
"""

import seaborn as sns

def distplot(data, x):
    """
    Recreates the (deprecated) sns.distplot function
    """
    sns.histplot(data=data, x=x, kde=True, stat='density', linewidth=0)
