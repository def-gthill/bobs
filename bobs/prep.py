"""
Data preparation and cleaning functions
"""

def split_y(df, y_column):
    """
    Splits the y column out of a dataframe containing
    both x and y columns.
    
    Returns a dataframe containing the x columns
    and a dataframe containing only the y column
    """
    return df.drop(y_column, axis=1), df[[y_column]]

