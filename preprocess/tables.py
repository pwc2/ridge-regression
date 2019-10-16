"""
    File name: tables.py
    Author: Patrick Cummings
    Date created: 10/12/2019
    Date last modified: 10/14/2019
    Python Version: 3.7
"""

import pathlib
import pickle

import numpy as np
import pandas as pd


def num_summary_csv(pickled_df):
    """Create summary table for numeric features from a DataFrame.

    Loads cleaned pickled DataFrame, produces summary table for numeric features as .csv file.

    Args:
        pickled_df: A pickled pandas DataFrame object.

    Returns:
        None
    """
    my_path = pathlib.Path(__file__).parent.resolve()
    out_path = my_path.joinpath('../tables').resolve()
    pkl_path = my_path.joinpath('../data', pickled_df).resolve()

    with open(pkl_path, 'rb') as pkl:
        df = pickle.load(pkl)

    filename = pkl_path.stem
    output_file = out_path.joinpath(filename + '_num_summary.csv')

    # Compute summary statistics, drop unnecessary ones, round, and output to .csv file
    numeric_summary = df.describe(percentiles=[], include=[np.number]).drop(labels=['count', '50%'], axis=0)
    numeric_summary.to_csv(output_file, float_format='%.2f')


def cat_summary_csv(pickled_df):
    """Create summary table for categorical features from a DataFrame.

    Loads cleaned pickled DataFrame, produces table of percentages for categorical features as a .csv file.
    Converts numerical features that should be categorical to categorical. Not pretty, but it works.

    Args:
        pickled_df: A pickled pandas DataFrame object.

    Returns:
        None
    """
    my_path = pathlib.Path(__file__).parent.resolve()
    out_path = my_path.joinpath('../tables').resolve()
    pkl_path = my_path.joinpath('../data', pickled_df).resolve()

    with open(pkl_path, 'rb') as pkl:
        df = pickle.load(pkl)

    filename = pkl_path.stem
    output_file = out_path.joinpath(filename + '_cat_summary.csv')

    convert = {'waterfront': 'category',
               'grade': 'category',
               'condition': 'category',
               'month': 'int64',
               'year': 'int64',
               'day': 'int64'}

    df = df.astype(convert)

    # Get proportions (value_counts) individually for the three categorical features
    waterfront_vc = pd.DataFrame(df['waterfront'].value_counts(normalize=True, sort=False) * 100).reset_index()
    waterfront_vc.columns = ['waterfront', 'percentage']

    condition_vc = pd.DataFrame(df['condition'].value_counts(normalize=True, sort=False) * 100).reset_index()
    condition_vc.columns = ['condition', 'percentage']

    grade_vc = pd.DataFrame(df['grade'].value_counts(normalize=True, sort=False) * 100).reset_index()
    grade_vc.columns = ['grade', 'percentage']

    # Join individual value counts back together in two separate joins
    int_join = grade_vc.join(condition_vc, how='outer', lsuffix='_grade', rsuffix='_condition')
    cat_table = int_join.join(waterfront_vc, how='outer', lsuffix='_condition', rsuffix='_waterfront')

    # Needed this to get rid of the NaN's that are produced in the categorical features during the joins
    cat_table['condition'] = cat_table['condition'].cat.add_categories('').fillna('')
    cat_table['waterfront'] = cat_table['waterfront'].cat.add_categories('').fillna('')
    cat_table.rename(columns={'percentage': 'percentage_waterfront'}, inplace=True)

    # Also needed to get rid of NaN's in the percentage columns from the joins
    cat_table['percentage_condition'] = cat_table['percentage_condition'].fillna(0).replace(0, '')
    cat_table['percentage_waterfront'] = cat_table['percentage_waterfront'].fillna(0).replace(0, '')

    cat_table.to_csv(output_file, float_format='%.2f')
