"""
    File name: gradient_descent.py
    Author: Patrick Cummings
    Date created: 10/12/2019
    Date last modified: 10/15/2019
    Python Version: 3.7

    Contains functions for training and evaluating linear regression model.
    calc_predictions() to calculate predictions on test or validation sets.
    calc_sse() to calculate sum-of-squared error on predictions.
    calc_gradient() to calculate the gradient of the weights during training.
    gradient_descent() to update weights during training.
"""

import numpy as np


def calc_predictions(x, weights):
    """Calculates predicted values using given weights.

    Args:
        x (ndarray): (n x m) ndarray of n observations on m features
        weights (ndarray): (1 x m) ndarray of m weights from learned model.

    Returns:
        list of float: List of predicted values.
    """
    weights = np.array(weights, dtype=np.float64)
    predictions = []
    for i in range(np.size(x, axis=0)):
        pred = float(weights.dot(x[i]))
        predictions.append(pred)
    return predictions


def calc_sse(x, y, weights, lam):
    """Calculate sum-of-squared errors (SSE) for predictions with L2 (quadratic) regularizer.

    Args:
        x (ndarray): (n, m) ndarray of n observations on m features.
        y (ndarray): (, n) or (n x 1) ndarray of n true response values for observations.
        weights (ndarray): (, m) ndarray of m weights from learned model.
        lam (float): regularization parameter.

    Returns:
        float: Sum of squared errors for predictions.
    """
    weights = np.array(weights, dtype=np.float64)
    y_pred = calc_predictions(x, weights)
    sse = 0
    for i in range(np.size(x, axis=0)):
        sse += (y[i] - y_pred[i]) ** 2 + (lam * weights.dot(weights) ** 2)
    return float(sse)


def calc_gradient(x, y, weights):
    """Calculates gradient for SSE loss function, without L2 regularization term.

    Args:
        x (ndarray): (n, m) ndarray of n observations on m features.
        y (ndarray): (, n) or (n, ) ndarray of n true response values for observations.
        weights(ndarray) : (, m) ndarray of m weights from learned model.

    Returns:
        grad (ndarray): Returns (, m) ndarray of calculated gradient.
    """
    grad = np.zeros(np.size(x, axis=1), dtype=np.float64)
    grad[0] = (x.dot(weights) - y).sum()
    for j in range(1, len(grad)):
        grad[j] = ((x.dot(weights) - y) * x[:, j]).sum()
    return grad


def gradient_descent(grad, weights, rate, lam):
    """Updates weights based on the gradient, includes L2 regularizer for non-intercept weights.

    Args:
        grad: (, m) ndarray of previous gradient computed for weights.
        weights: (, m) ndarray of m weights from model in training.
        rate: learning rate.
        lam: regularization parameter.

    Returns:
        weights (ndarray): Returns (, m) ndarray of updated weights.
    """
    weights[0] -= rate * grad[0]
    for j in range(1, len(weights)):
        weights[j] -= rate * (grad[j] + lam * weights[j])
    return weights
