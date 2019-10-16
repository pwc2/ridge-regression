"""
    File name: run_part3.py
    Author: Patrick Cummings
    Date created: 10/13/2019
    Date last modified: 10/15/2019
    Python Version: 3.7

    Used to run linear regression without L2 regularization penalty for given learning rates.
    Trains model on training set, makes predictions and calculates sum-of-squared error on validation set.
    Using non-normalized data sets here.
    Outputs model results to /model_output/part_3.
"""

import json
import pathlib

import numpy as np

from models.linear_model import LinearModel

print('Part 3: Initializing training without regularization, without normalizing features.')

# Ignore overflows from learning rates with exploding gradient
np.seterr(all='ignore')

# Training - Part 3, exploring training with non-normalized data.

rates = [10 ** -15, 10 ** -9, 10 ** -6, 10 ** -3, 0, 1]
for rate in rates:
    model = LinearModel(train='data/PA1_train_one_hot.pkl',
                        validation='data/PA1_dev_one_hot.pkl',
                        test='data/PA1_test_one_hot.pkl',
                        target='price',
                        rate=rate,
                        lam=0,
                        eps=0.5)

    learned_model = model.train_model(max_iter=10000)

    print('Training complete.')

    # Save output for learned model to .json file
    train_filename = 'rate_' + str('{:.0E}'.format(rate)) + '_train.json'
    my_path = pathlib.Path('model_output', 'part_3', train_filename)
    train_path = pathlib.Path(__file__).parent.resolve().joinpath(my_path)

    # Create directory if doesn't exist
    if not pathlib.Path(train_path).exists():
        pathlib.Path(train_path).mkdir()
    with open(train_path, 'w') as f:
        json.dump(learned_model, f, indent=4)

    print('Calculating predictions on validation set...')

    # Grab weights to input to prediction method
    weights = learned_model['weights']
    predictions = model.predict_validation(weights)

    # Output for predictions and SSE for validation set
    dev_filename = 'rate_' + str('{:.0E}'.format(rate)) + '_dev.json'
    dev_path = train_path.with_name(dev_filename)

    # Create directory if doesn't exist
    if not pathlib.Path(dev_path).exists():
        pathlib.Path(dev_path).mkdir()
    with open(dev_path, 'w') as f:
        json.dump(predictions, f, indent=4)

    print('Predictions complete.')

print('Part 3 complete.\n')
