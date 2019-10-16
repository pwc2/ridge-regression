"""
    File name: run_part1.py
    Author: Patrick Cummings
    Date created: 10/13/2019
    Date last modified: 10/15/2019
    Python Version: 3.7

    Used to run linear regression without L2 regularization penalty for given learning rates.
    Trains model on training set, makes predictions and calculates sum-of-squared error on validation set.
    Using normalized data sets here.
    Outputs model results to /model_output/part_1.
"""

import json
import pathlib

import numpy as np

from models.linear_model import LinearModel

# Ignore overflows from learning rates with exploding gradient
np.seterr(all='ignore')

# Training - Part 1, exploring learning rates.

# Train model using each learning rate, save each training output to model_output/part_1 folder.
rates = [10 ** -x for x in range(8)]
for rate in rates:
    model = LinearModel(train='data/PA1_train_norm.pkl',
                        validation='data/PA1_dev_norm.pkl',
                        test='data/PA1_test_norm.pkl',
                        target='price',
                        rate=rate,
                        lam=0,
                        eps=0.5)

    learned_model = model.train_model(max_iter=10000)

    # Save output for learned model to .json file
    train_filename = 'rate_' + str('{:.0E}'.format(rate)) + '_train.json'
    my_path = pathlib.Path('model_output', 'part_1', train_filename)
    train_path = pathlib.Path(__file__).parent.resolve().joinpath(my_path)
    with open(train_path, 'w') as f:
        json.dump(learned_model, f, indent=4)

# Get predictions and SSE on validation set for rates that didn't explode
working_rates = [10 ** -x for x in range(5, 8)]
for working_rate in working_rates:
    model = LinearModel(train='data/PA1_train_norm.pkl',
                        validation='data/PA1_dev_norm.pkl',
                        test='data/PA1_test_norm.pkl',
                        target='price',
                        rate=working_rate,
                        lam=0,
                        eps=0.5)

    # Open learned model to extract training weights
    train_filename = 'rate_' + str('{:.0E}'.format(working_rate)) + '_train.json'
    my_path = pathlib.Path('model_output', 'part_1', train_filename)
    train_path = pathlib.Path(__file__).parent.resolve().joinpath(my_path)
    with open(train_path) as f:
        learned_model = json.load(f)

    # Grab weights to input to prediction method
    weights = learned_model['weights']
    predictions = model.predict_validation(weights)

    # Output for predictions and SSE for validation set
    dev_filename = 'rate_' + str('{:.0E}'.format(working_rate)) + '_dev.json'
    dev_path = train_path.with_name(dev_filename)
    with open(dev_path, 'w') as f:
        json.dump(predictions, f, indent=4)
