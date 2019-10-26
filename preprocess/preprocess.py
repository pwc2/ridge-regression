"""
    File name: preprocess.py
    Author: Patrick Cummings
    Date created: 10/12/2019
    Date last modified: 10/18/2019
    Python Version: 3.7

    Used to pre-process data.
    clean() function to clean data by splitting date feature into month, day, and year. Drops ID and date columns.
 """

import pickle
from pathlib import Path

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

    # Remove ID, parse month, day, year into individual columns.
    df.drop(labels='id', axis=1, inplace=True)
    df = df.assign(month=df['date'].map(lambda x: x.split('/')[0]),
                   day=df['date'].map(lambda x: x.split('/')[1]),
                   year=df['date'].map(lambda x: x.split('/')[2]))
    df.drop(labels='date', axis=1, inplace=True)

    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    df['year'] = df['year'].astype(int)

    # Ensure all columns in same order, then reset first column to dummy.
    df.sort_index(axis=1, inplace=True)
    df = df.set_index('dummy').reset_index()

    # Get path to output clean pickled DataFrame.
    my_path = Path(input_file).resolve()
    out_path = my_path.with_suffix('.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
    return
