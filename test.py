"""
    File name: test.py
    Author: Patrick Cummings
    Date created: 10/13/2019
    Date last modified: 10/15/2019
    Python Version: 3.7

    Used to test instance of LinearModel() class.
"""

import pprint

from models.linear_model import LinearModel

pp = pprint.PrettyPrinter()

model = LinearModel(train='data/PA1_train.pkl',
                    validation='data/PA1_dev.pkl',
                    test='data/PA1_test.pkl',
                    target='price',
                    rate=1e-05,
                    lam=0,
                    eps=0.5,
                    normalize=True)

names = model.weight_labels
learned_model = model.train_model(10)
val_predictions = model.predict_validation(learned_model['weights'])['predictions']
test_predictions = model.predict_test((learned_model['weights']))['predictions']

# pp.pprint(learned_model)

# print(dict(zip(names, learned_model['weights'])))
pp.pprint(val_predictions[:10])
pp.pprint(test_predictions[:10])
