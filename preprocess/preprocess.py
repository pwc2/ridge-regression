"""
    File name: preprocess.py
    Author: Patrick Cummings
    Date created: 10/12/2019
    Date last modified: 10/18/2019
    Python Version: 3.7

    Used to pre-process data.
    clean() function to clean data by splitting date feature into month, day, and year. Drops ID and date columns.
 """

import pathlib
import pickle

import pandas as pd


def clean(input_file):
    """Remove ID feature, and split date feature into month, day, and year, then drop original date feature.

    Reads in a .csv file.

    Args:
        input_file (str): String with filepath to .csv file with historic data on houses sold between May 2014 to May
        2015.

    Returns:
        None
    """
    df = pd.read_csv(input_file)

    # Remove ID, parse month, day, year into individual columns and
    df.drop(labels='id', axis=1, inplace=True)
    df = df.assign(month=df['date'].map(lambda x: x.split('/')[0]),
                   day=df['date'].map(lambda x: x.split('/')[1]),
                   year=df['date'].map(lambda x: x.split('/')[2]))
    df.drop(labels='date', axis=1, inplace=True)

    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    df['year'] = df['year'].astype(int)

    # Ensure all columns in same order, then reset first column to dummy
    df.sort_index(axis=1, inplace=True)
    df = df.set_index('dummy').reset_index()

    # Get path to output clean pickled DataFrame
    my_path = pathlib.Path(input_file).resolve()
    out_path = my_path.with_suffix('.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
    return


# def normalize(pickled_df, train_max, train_min, target):
#     """Normalize features to (0, 1] scale based on the training data. Target not normalized.
#
#         Use on .pkl file produced by calling clean() on the .csv files for training, test, and validation.
#
#         Args:
#             pickled_df: String with filepath to cleaned pickled DataFrame.
#             train_max (ndarray): Maximum values for features in training data.
#             train_min (ndarray): Minimum values in features training data.
#             target (str): Response variable.
#         Returns:
#             normdf (DataFrame): DataFrame with normalized features, target remains non-normalized.
#         """
#     with open(pickled_df, 'rb') as pkl:
#         df = pickle.load(pkl)
#
#     # Remove target from normalization
#     target_col = None
#     if target in df:
#         target_col = df[target]
#         df.drop(labels=target, axis=1, inplace=True)
#
#     # Get range for (0,1) normalization
#     train_range = train_max - train_min
#
#     # Normalize data set based on training set, reassign 1 to dummy, and make sure dummy is first column
#     norm_df = df.sub(train_min).div(train_range).assign(dummy=1).set_index('dummy').reset_index()
#
#     # Add back in target column for prediction if necessary, drop blank column from test set
#     if target_col is not None:
#         norm_df[target] = target_col
#     else:
#         norm_df.drop(target, axis=1, inplace=True)
#
#     # Save normalized DataFrame as pickled object if it doesn't exist yet
#     my_path = pathlib.Path(pickled_df).resolve()
#     filename = my_path.stem
#     out_path = my_path.with_name(filename + '_norm.pkl')
#     if not pathlib.Path(out_path).exists():
#         with open(out_path, 'wb') as f:
#             pickle.dump(norm_df, f, pickle.HIGHEST_PROTOCOL)
#     return norm_df
