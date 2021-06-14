import unittest

import pandas as pd
import pandas.testing as pt

from bobs import prep


class TestPrep(unittest.TestCase):
    def test_split_y(self):
        df = pd.DataFrame(
            {
                'Foo': [1, 2, 3],
                'Bar': [2, 4, 6],
                'Baz': ['y', 'n', 'y'],
            },
            index=['a1', 'b2', 'c3'],
        )
        x, y = prep.split_y(df, 'Baz')
        pt.assert_frame_equal(
            x, pd.DataFrame(
                {
                    'Foo': [1, 2, 3],
                    'Bar': [2, 4, 6],
                },
                index=['a1', 'b2', 'c3'],
            )
        )
        pt.assert_frame_equal(
            y, pd.DataFrame(
                {
                    'Baz': ['y', 'n', 'y'],
                },
                index=['a1', 'b2', 'c3'],
            )
        )
    
    def test_pandas_column_transformer(self):
        """
        This example comes from Johannes Haupt, in
        the same blog post as the implementation of
        PandasColumnTransformer.
        """
        from sklearn.datasets import fetch_openml

        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        x, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
        
        numeric_features = ['age', 'fare']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]
        )

        categorical_features = ['embarked', 'sex', 'pclass']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = prep.PandasColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        transformed = preprocessor.fit_transform(x)
        self.assertEqual(
            list(transformed.columns),
            [
                'age', 'fare',
                'embarked_C', 'embarked_Q', 'embarked_S', 'embarked_missing',
                'sex_female', 'sex_male',
                'pclass_1.0', 'pclass_2.0', 'pclass_3.0',
            ]
        )
