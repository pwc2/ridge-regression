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

model = LinearModel(train='data/PA1_train_norm.pkl',
                    validation='data/PA1_dev_norm.pkl',
                    test='data/PA1_test_norm.pkl',
                    target='price',
                    rate=10 ** -5,
                    lam=0,
                    eps=0.5)

names = model.get_weight_labels()
learned_model = model.train_model(10000)
val_predictions = model.predict_validation(learned_model['weights'])['predictions']
test_predictions = model.predict_test((learned_model['weights']))['predictions']

print(dict(zip(names, learned_model['weights'])))
pp.pprint(val_predictions[:10])
pp.pprint(test_predictions[:10])
