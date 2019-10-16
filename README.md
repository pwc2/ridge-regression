# ridge-regression

ridge-regression is an implementation of linear regression with L2 regularization that uses batch gradient descent to optimize weights.

## Usage

```python
from models.linear_model import LinearModel

model = LinearModel(train='data/PA1_train_norm.pkl', # Path to training set
                    validation='data/PA1_dev_norm.pkl', # Path to validation set
                    test='data/PA1_test_norm.pkl', # Path to test set
                    target='price', # Target for prediction
                    rate=10 ** -5, # Learning rate for gradient descent
                    lam=0, # Regularization penalty
                    eps=0.5) # Stopping condition for gradient descent

names = model.get_weight_labels() # Extract labels for weights
learned_model = model.train_model(max_iter=10000) # Train model
val_predictions = model.predict_validation(learned_model['weights'])['predictions'] # Get predictions on validation set
test_predictions = model.predict_test((learned_model['weights']))['predictions'] # Get predictions on test set
```

