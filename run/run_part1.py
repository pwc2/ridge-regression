"""
    File name: run_part1.py
    Author: Patrick Cummings
    Date created: 10/13/2019
    Date last modified: 10/18/2019
    Python Version: 3.7

    Used to run linear regression without L2 regularization penalty for given learning rates.
    Trains model on training set, makes predictions and calculates sum-of-squared error on validation set.
    Using normalization.
    Outputs model results to /model_output/part_1.
"""

import json
from pathlib import Path

import numpy as np

from models.linear_model import LinearModel

print('Part 1: Initializing training without regularization.\n')

# Ignore overflows from learning rates with exploding gradient.
np.seterr(all='ignore')

# Training - Part 1, exploring learning rates without regularization.

lam = 0
rates = [10 ** -x for x in range(8)]
for rate in rates:
    model = LinearModel(train='data/PA1_train.pkl',
                        validation='data/PA1_dev.pkl',
                        test='data/PA1_test.pkl',
                        target='price',
                        rate=rate,
                        lam=lam,
                        eps=2,
                        normalize=True)

    learned_model = model.train_model(max_iter=10000)

    if learned_model['exploding'] is False and learned_model['convergence'] is True:
        print('Training complete.')

    # Save output for learned model to .json file.
    filename = 'rate_' + str('{:.0E}'.format(rate)) + '_lam_' + str('{:.0E}'.format(lam))
    my_path = Path('..', 'model_output', 'part_1', filename + '_train.json')
    train_path = Path(__file__).parent.resolve().joinpath(my_path)

    # Make output directory if doesn't exist.
    output_dir = train_path.parent.resolve()
    if not Path(output_dir).exists():
        Path(output_dir).mkdir()

    with open(train_path, 'w') as f:
        json.dump(learned_model, f, indent=4)

    # If gradient didn't explode, get predictions on validation set.
    if learned_model['exploding'] is False:
        print('Calculating predictions on validation set...')

        # Grab weights to input to prediction method.
        weights = learned_model['weights']
        predictions = model.predict_validation(weights)

        # Output for predictions and SSE for validation set.
        dev_filename = filename + '_dev.json'
        dev_path = train_path.with_name(dev_filename)

        with open(dev_path, 'w') as f:
            json.dump(predictions, f, indent=4)

        print('Predictions complete.\n')

print('Part 1 complete.\n')
