"""
Data preparation and cleaning functions
"""

def transform_pd(sk_transformer, df, columns=None):
    """
    Applies the specified scikit-learn transformer to
    the specified Pandas dataframe. The result is
    a dataframe with the same labels as ``df``.
    
    If ``columns`` is specified, it must be a list of
    column names to transform. Only those columns
    will be changed.
    """
    if columns is None:
        columns = df.columns
    
    result = df.copy()
    result[columns] = sk_transformer.fit_transform(df[columns])
    return result


def inverse_transform_pd(sk_transformer, df, columns=None):
    """
    Applies the inverse of the specified (fitted)
    scikit-learn transformer to
    the specified Pandas dataframe. The result is
    a dataframe with the same labels as ``df``.
    
    If ``columns`` is specified, it must be a list of
    column names to transform. Only those columns
    will be changed.
    """
    if columns is None:
        columns = df.columns
    
    result = df.copy()
    result[columns] = sk_transformer.inverse_transform(df[columns])
    return result
    
