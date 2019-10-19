"""
    File name: run_part2.py
    Author: Patrick Cummings
    Date created: 10/13/2019
    Date last modified: 10/18/2019
    Python Version: 3.7

    Run linear regression with L2 regularization with grid search over given learning rates and penalties.
    Trains models on training set, makes predictions and calculates sum-of-squared errors on validation set.
    Using normalization.
    Outputs model results to /model_output/part_2.
"""

import json
import pathlib

import numpy as np

from models.linear_model import LinearModel

print('Part 2: Initializing training with regularization.\n')

# Ignore overflows from learning rates with exploding gradient
np.seterr(all='ignore')

# Training - Part 2, adjusting regularization parameter to investigate effect on the model.

rates = [10 ** -x for x in range(5, 8)]
# rates = [10 ** -x for x in range(8)]
lambdas = sorted([10 ** x for x in range(-3, 3)] + [0])[4:]
for rate in rates:
    for lam in lambdas:

        model = LinearModel(train='data/PA1_train.pkl',
                            validation='data/PA1_dev.pkl',
                            test='data/PA1_test.pkl',
                            target='price',
                            rate=rate,
                            lam=lam,
                            eps=0.5,
                            normalize=True)

        learned_model = model.train_model(max_iter=100000)

        print('Training complete.\n')

        # Save output for learned model to .json file
        train_filename = 'rate_' + str('{:.0E}'.format(rate)) + '_lam_' + str('{:.0E}'.format(lam)) + '_train.json'
        my_path = pathlib.Path('..', 'model_output', 'part_2', train_filename)
        train_path = pathlib.Path(__file__).parent.resolve().joinpath(my_path)

        # Make output directory if doesn't exist
        output_dir = train_path.parent.resolve()
        if not pathlib.Path(output_dir).exists():
            pathlib.Path(output_dir).mkdir()

        with open(train_path, 'w') as f:
            json.dump(learned_model, f, indent=4)

        print('Calculating predictions on validation set...')

        # Grab weights to input to prediction method
        weights = learned_model['weights']
        predictions = model.predict_validation(weights)

        # Output for predictions and SSE for validation set
        dev_filename = 'rate_' + str('{:.0E}'.format(rate)) + '_lam_' + str('{:.0E}'.format(lam)) + '_dev.json'
        dev_path = train_path.with_name(dev_filename)

        with open(dev_path, 'w') as f:
            json.dump(predictions, f, indent=4)

        print('Predictions complete.\n')

print('Part 2 complete.\n')
