"""
    File name: preprocess.py
    Author: Patrick Cummings
    Date created: 10/12/2019
    Date last modified: 10/15/2019
    Python Version: 3.7

    Used to pre-process data.
    clean() function to clean data by splitting date feature into month, day, and year. Drops ID and date columns.
    one_hot_encode() function to one-hot-encode categorical features from cleaned data.
    normalize() function to normalize all features to range zero to one based on training set.
 """

import pathlib
import pickle

import pandas as pd


def clean(input_file):
    """Remove ID feature, and split date feature into month, day, and year, then drop original date feature.

    Reads in a .csv file.

    Args:
        input_file: String with filepath to .csv file with historic data on houses sold between May 2014 to May 2015.

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

    # Get path to output cleaned pickled DataFrame
    my_path = pathlib.Path(input_file).resolve()
    out_path = my_path.with_suffix('.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


def one_hot_encode(pickled_df):
    """One-hot encode categorical features for this regression task. Use on pickled file after calling clean().

    No normalization applied.

    Args:
        pickled_df: String with filepath to cleaned pickle file.

    Returns:
        None
    """
    with open(pickled_df, 'rb') as pkl:
        df = pickle.load(pkl)

    df = pd.get_dummies(df, columns=['waterfront', 'grade', 'condition'], drop_first=True)

    # Drop reference category that somehow shows up in validation set
    if 'grade_4' in df:
        df.drop(labels='grade_4', axis=1, inplace=True)

    # Ensure all columns in same order
    df.sort_index(axis=1, inplace=True)

    # Move dummy (intercept) to index
    df = df.set_index('dummy').reset_index()

    # Get path to output one-hot-encoded pickled DataFrame
    my_path = pathlib.Path(pickled_df).resolve()
    filename = my_path.stem
    out_path = my_path.with_name(filename + '_one_hot.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


def normalize(train, test, validation):
    """Normalize features for training, test, and validation sets based on the training data.

    Use after calling clean() on the .csv files for training, test, and validation.

    Args:
        train: String with filepath to cleaned pickle file for training data.
        test: String with filepath to cleaned pickle file for test data.
        validation: String with filepath to cleaned pickle file for validation data.

    Returns:
        None
    """
    with open(train, 'rb') as pkl:
        df_train = pickle.load(pkl)
    with open(test, 'rb') as pkl:
        df_test = pickle.load(pkl)
    with open(validation, 'rb') as pkl:
        df_validation = pickle.load(pkl)

    # One-hot encode categorical features
    df_train = pd.get_dummies(df_train, columns=['waterfront', 'grade', 'condition'], drop_first=True)
    df_test = pd.get_dummies(df_test, columns=['waterfront', 'grade', 'condition'], drop_first=True)
    df_validation = pd.get_dummies(df_validation, columns=['waterfront', 'grade', 'condition'], drop_first=True)

    # Remove target before normalization from train and validation sets
    train_target = df_train['price']
    validation_target = df_validation['price']
    df_train.drop(labels='price', axis=1, inplace=True)
    df_validation.drop(labels='price', axis=1, inplace=True)

    # Normalize all three sets using training min/max for each feature
    norm_df_train = (df_train - df_train.min()) / (df_train.max() - df_train.min())
    norm_df_test = (df_test - df_train.min()) / (df_train.max() - df_train.min())
    norm_df_validation = (df_validation - df_train.min()) / (df_train.max() - df_train.min())

    # Drop reference category that somehow shows up in validation set
    norm_df_validation.drop(labels='grade_4', axis=1, inplace=True)

    # Drop price if in test set
    if 'price' in norm_df_test:
        norm_df_test.drop(labels='price', axis=1, inplace=True)

    # Add target back to train and validation sets
    norm_df_train['price'] = train_target
    norm_df_validation['price'] = validation_target

    # Ensure columns all in same order
    norm_df_train.sort_index(axis=1, inplace=True)
    norm_df_test.sort_index(axis=1, inplace=True)
    norm_df_validation.sort_index(axis=1, inplace=True)

    # Re-assign 1 to dummy column after normalization
    for df in [norm_df_train, norm_df_test, norm_df_validation]:
        df['dummy'] = 1

    # Move dummy (intercept) to index
    norm_df_train = norm_df_train.set_index('dummy').reset_index()
    norm_df_test = norm_df_test.set_index('dummy').reset_index()
    norm_df_validation = norm_df_validation.set_index('dummy').reset_index()

    # Pickle normalized data sets
    for name, norm_df in [(train, norm_df_train), (test, norm_df_test), (validation, norm_df_validation)]:
        my_path = pathlib.Path(name).resolve()
        filename = my_path.stem
        out_path = my_path.with_name(filename + '_norm.pkl')

        with open(out_path, 'wb') as f:
            pickle.dump(norm_df, f, pickle.HIGHEST_PROTOCOL)
