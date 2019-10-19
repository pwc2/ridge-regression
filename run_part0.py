"""
    File name: run_part0.py
    Author: Patrick Cummings
    Date created: 10/13/2019
    Date last modified: 10/18/2019
    Python Version: 3.7

    Preprocesses data to create cleaned DataFrames, which are used in LinearModel constructor.
    Cleaned DataFrames saved as .pkl files in /data.
"""

from preprocess import preprocess, tables

# Pre-processing - Part 0.

print('\nPart 0: Initializing preprocessing...')

# Clean .csv files
preprocess.clean('data/PA1_train.csv')
preprocess.clean('data/PA1_test.csv')
preprocess.clean('data/PA1_dev.csv')

# Create tables
tables.num_summary_csv('PA1_train.pkl')
tables.cat_summary_csv('PA1_train.pkl')

print('Completed preprocessing.\n')
