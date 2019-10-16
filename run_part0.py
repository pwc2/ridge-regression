"""
    File name: run_part0.py
    Author: Patrick Cummings
    Date created: 10/13/2019
    Date last modified: 10/15/2019
    Python Version: 3.7

    Pre-processes data to create cleaned DataFrames, which are then used to create one-hot-encoded non-normalized
    DataFrames, as well as one-hot-encoded normalized DataFrames. All saved as .pkl files in /data.
"""

from preprocess import preprocess, tables

# Pre-processing - Part 0.

# Clean
preprocess.clean('data/PA1_train.csv')
preprocess.clean('data/PA1_test.csv')
preprocess.clean('data/PA1_dev.csv')

# Create tables
tables.num_summary_csv('PA1_train.pkl')
tables.cat_summary_csv('PA1_train.pkl')

# Normalize, for use in parts 1 and 2
preprocess.normalize('data/PA1_train.pkl', 'data/PA1_test.pkl', 'data/PA1_dev.pkl')

# Create non-normalized one-hot encoded data sets, for use in part 3
preprocess.one_hot_encode('data/PA1_train.pkl')
preprocess.one_hot_encode('data/PA1_dev.pkl')
preprocess.one_hot_encode('data/PA1_test.pkl')
