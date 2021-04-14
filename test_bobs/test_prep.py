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
        
