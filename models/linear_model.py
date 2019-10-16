"""
    File name: linear_model.py
    Author: Patrick Cummings
    Date created: 10/13/2019
    Date last modified: 10/15/2019
    Python Version: 3.7
"""

import pickle

import numpy as np
import progressbar

from models.gradient_descent import calc_sse, calc_predictions, calc_gradient, gradient_descent


class LinearModel:
    """Class to create linear model with L2 (ridge) penalty.

    """

    def __init__(self, train, validation, test, target, rate, lam, eps):
        """Constructs a linear model with supplied training data, test data, learning rate, regularization parameter,
        and convergence threshold. Uses L2 regularization with batch gradient descent to optimize weights.

        Args:
            train (str): Path to pickled DataFrame with training examples.
            validation (str): Path to pickled DataFrame with validation examples.
            test (str): Path to pickled DataFrame with testing examples.
            target (str): A string with column name for response feature.
            rate (float): The learning rate for gradient descent.
            lam (float): The regularization parameter.
            eps (float): The convergence criterion for gradient descent.
          """
        with open(train, 'rb') as train_pkl:
            self.train = pickle.load(train_pkl)
        with open(validation, 'rb') as validation_pkl:
            self.validation = pickle.load(validation_pkl)
        with open(test, 'rb') as test_pkl:
            self.test = pickle.load(test_pkl)
        self.target = target
        self.rate = rate
        self.lam = lam
        self.eps = eps
        self.train_labels = self.train[target]
        self.validation_labels = self.validation[target]

        # Include assertion to check that columns for test, validation, and test in same order.
        check_train = self.train.drop(target, axis=1).columns.tolist()
        check_validation = self.validation.drop(target, axis=1).columns.tolist()
        check_test = self.test.columns.tolist()
        assert check_train == check_validation == check_test, 'Columns not in same order for train, val, test.'

    def get_weight_labels(self):
        """Retrieves labels for model weights.

        Returns:
            list of str: List of weight labels.
        """
        x = self.train.drop(self.target, axis=1)
        return x.columns.tolist()

    def train_model(self, max_iter):
        """Trains model with batch gradient descent to optimize weights.

        Calls calc_sse(), calc_gradient(), gradient_descent() from models.gradient_descent.py.

        Args:
            max_iter (int): Maximum number of iterations before terminating training.

        Returns:
            results (dict): Dictionary with lam (regularization parameter), learn_rate (learning rate), eps (epsilon
            for convergence), iterations (number of iterations), convergence (T or F), exploding (if gradient
            explodes), labeled_weights (dict with {labels : weights}), weights (nd array of optimized weights),
            and sse (list with SSE for each iteration). Also includes gradient_norm_diff (normalized difference in
            gradient between iterations).
        """

        # x is training observations, y is training labels
        x = self.train.drop(self.target, axis=1).to_numpy(dtype=np.float64)
        y = self.train_labels.to_numpy(dtype=np.float64)
        rate = self.rate
        lam = self.lam
        eps = self.eps
        weight_labels = self.train.drop(self.target, axis=1).columns.tolist()

        print('Initializing training...')

        # Initialize random weights in sampled from uniform [0, 1) distribution
        weights = np.random.rand(np.size(x, axis=1))

        print('Lambda = ' + str(lam) + ', rate = ' + str(rate) + ', epsilon = ' + str(eps) + '.')

        # Shape of x[i] is (33, 1), shape of y[i] is (1), shape of weights*x[i] - y[i] is (1), shape of grad is (33, 1).
        print('Optimizing weights...')

        # Included a progress bar for iteration progress, list for SSE's, indicators for gradient and convergence.
        bar = progressbar.ProgressBar()
        iter_count = 0
        sse_list = []
        norm_list = []
        exploding_grad = False
        converge = False

        # Initialize empty gradient for stopping condition norm calculation
        previous_grad = np.zeros(np.size(x, axis=1), dtype=np.float64)

        # Perform batch gradient descent to optimize weights
        for iteration in bar(range(max_iter)):
            # Calculate gradient and update weights
            current_grad = calc_gradient(x, y, weights)
            weights = gradient_descent(current_grad, weights, rate, lam)

            # Calculate sum of squared error for each iteration to store in list
            sse_list.append(calc_sse(x, y, weights, lam))

            # Calculate norm of difference between current and previous gradients for each iteration and store
            grad_diff = current_grad - previous_grad
            norm_diff = np.sqrt(grad_diff.dot(grad_diff))
            norm_list.append(norm_diff)

            # Set previous_grad to current_grad for next iteration
            previous_grad = current_grad
            iter_count += 1

            # Check for divergence with the norm of the gradient to see if exploding
            if np.isinf(norm_diff):
                print('\nGradient exploding.\n')
                exploding_grad = True
                break

            # Check for convergence using the norm of the difference in current and previous gradients
            if norm_diff < eps:
                print('\nConvergence achieved with epsilon = ' + str(eps) + ' in ' + str(iteration) + ' iterations.')
                converge = True
                break

        # If we haven't converged by this point might as well stop and figure out why
        if iter_count == max_iter:
            print('Maximum iterations reached without convergence.\n')

        labeled_weights = dict(zip(weight_labels, weights.tolist()))

        results = {'lam': lam,
                   'learn_rate': rate,
                   'epsilon': eps,
                   'iterations': iter_count,
                   'convergence': converge,
                   'exploding': exploding_grad,
                   'labeled_weights': labeled_weights,
                   'weights': weights.tolist(),
                   'sse': sse_list,
                   'gradient_norm_diff': norm_list}
        return results

    def predict_validation(self, weights):
        """Generates predictions and sum-of-squared error calculations for validation data with labels.

        Uses validation set provided when instance of class is created.

        Args:
            weights (ndarray): (1, 33) ndarray (or list) of weights produced from training.

        Returns:
            results (dict): Dictionary with lam (regularization parameter), predictions (list of predictions),
            and SSE (SSE for predictions).
        """
        lam = self.lam
        x = self.validation.drop(self.target, axis=1).to_numpy(dtype=np.float64)
        y = self.validation_labels.to_numpy(dtype=np.float64)
        predictions = calc_predictions(x, weights)
        sse = calc_sse(x, y, weights, lam)
        results = {'lam': lam,
                   'SSE': sse,
                   'predictions': predictions}
        return results

    def predict_test(self, weights):
        """Generates predictions for unlabeled test data.

        Args:
            weights (ndarray): (1, 33) ndarray (or list) of weights produced from training.

        Returns:
            results (dict): Dictionary with lam (regularization parameter) and predictions (list of predictions).
        """
        lam = self.lam
        x = self.test.to_numpy(dtype=np.float64)
        predictions = calc_predictions(x, weights)
        results = {'lam': lam,
                   'predictions': predictions}
        return results
