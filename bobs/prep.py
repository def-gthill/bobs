"""
Data preparation and cleaning functions
"""

import numpy as np
import pandas as pd

from sklearn import base, compose, pipeline


def split_y(df, y_column):
    """
    Splits the y column out of a dataframe containing
    both x and y columns.
    
    Returns a dataframe containing the x columns
    and a dataframe containing only the y column
    """
    return df.drop(y_column, axis=1), df[[y_column]]


class ColumnAssigner(base.TransformerMixin):
    """
    A scikit-learn transformer that converts a numpy
    array into a Pandas dataframe with the specified
    column names. Useful for pipelines where later
    steps need column names (e.g. ColumnTransformer)
    but earlier steps strip the column names off
    (e.g. every single built-in transformer).
    """

    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, data, targets=None):
        return self

    def transform(self, data):
        return pd.DataFrame(data, columns=self.column_names)


class PandasColumnTransformer(base.TransformerMixin):
    """
    A scikit-learn transformer that acts exactly like
    a ColumnTransformer, but returns a Pandas dataframe
    with sensible column names.

    The constructor takes the same arguments as a
    ColumnTransformer.

    The implementation comes (with slight modifications to
    make the column names nicer) from this blog post by
    Johannes Haupt: https://johaupt.github.io/scikit-learn/tutorial/python/data%20processing/ml%20pipeline/model%20interpretation/columnTransformer_feature_names.html
    """

    def __init__(self, *args, **kwargs):
        self.transformer = compose.ColumnTransformer(*args, **kwargs)

    def fit(self, data, targets=None):
        self.transformer.fit(data, targets)
        self._column_names = self._get_column_names(self.transformer)
        return self

    def transform(self, data):
        transformed_values = self.transformer.transform(data)
        return pd.DataFrame(
            transformed_values,
            index=data.index,
            columns=self._column_names,
        )

    @classmethod
    def _get_column_names(cls, transformer):
        feature_names = []

        for name, sub_transformer, columns, _ in transformer._iter(fitted=True):
            if type(sub_transformer) == pipeline.Pipeline:
                # Recursive call on pipeline
                _names = cls._get_column_names_pipeline(sub_transformer, columns)
                # if pipeline has no transformer that returns names
                if len(_names) == 0:
                    _names = columns
                feature_names.extend(_names)
            else:
                feature_names.extend(cls.get_names(transformer, sub_transformer, columns))

        return feature_names

    @classmethod
    def _get_column_names_pipeline(cls, transformer, columns=None):
        for _, name, sub_transformer in transformer._iter():
            columns = cls.get_names(transformer, sub_transformer, columns)

        return columns

    @classmethod
    def get_names(cls, transformer, trans, columns):
        if trans == 'drop' or (
                hasattr(columns, '__len__') and not len(columns)):
            return []
        if trans == 'passthrough':
            if hasattr(transformer, '_df_columns'):
                if ((not isinstance(columns, slice))
                        and all(isinstance(col, str) for col in columns)):
                    return columns
                else:
                    return transformer._df_columns[columns]
            else:
                indices = np.arange(transformer._n_features)
                return [f'x{i}' for i in indices[columns]]
        if not hasattr(trans, 'get_feature_names'):
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if columns is None:
                return []
            else:
                return columns

        if columns is None:
            return trans.get_feature_names()
        else:
            feature_names = list(trans.get_feature_names())
            for i, col in enumerate(columns):
                for j, feature_name in enumerate(feature_names):
                    feature_names[j] = feature_name.replace(f'x{i}', col)
            return feature_names


class ColumnKeeper(base.TransformerMixin):
    """
    A scikit-learn transformer that keeps only the columns
    of the dataframe with the specified names.
    """

    def __init__(self, column_names):
        self.column_names = column_names

    # noinspection PyUnusedLocal
    def fit(self, data, targets=None):
        return self

    def transform(self, data):
        return data[self.column_names]


class ColumnDropper(base.TransformerMixin):
    """
    A scikit-learn transformer that drops the columns
    of the dataframe with the specified names.
    
    If the data passed to transform has different
    column names than the data passed to fit,
    the transformed dataframe will have the columns
    that the fit dataframe would have had if the
    columns had been dropped.
    
    For example, if a dataframe with the columns
    ['foo', 'bar', 'baz'] is passed to fit,
    and columns is ['bar'], then transform
    will always keep the 'foo' and 'baz' columns
    and only those columns, regardless of what
    the rest of the dataframe looks like.
    
    That way, if there are irrelevant columns
    in the training data and different irrelevant
    columns in the production data, the column
    dropper will keep the same set of columns
    from each.
    """

    def __init__(self, column_names):
        self.columns_to_drop = column_names

    # noinspection PyUnusedLocal
    def fit(self, data, targets=None):
        columns_to_drop_set = set(self.columns_to_drop)
        self.columns_to_keep = [
            column for column in data.columns
            if column not in columns_to_drop_set
        ]
        return self

    def transform(self, data):
        return data[self.columns_to_keep]
