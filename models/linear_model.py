"""
    File name: linear_model.py
    Author: Patrick Cummings
    Date created: 10/13/2019
    Date last modified: 10/18/2019
    Python Version: 3.7
"""

import pathlib
import pickle

import numpy as np
import progressbar

from models.gradient_descent import compute_sse, predict, compute_gradient, gradient_descent


class LinearModel:
    """Class to construct linear model object with L2 loss.

    """

    def __init__(self, train, validation, test, target, rate, lam, eps, normalize=True):
        """Constructs a linear model with supplied training data, test data, learning rate, regularization parameter,
        and convergence threshold. Uses L2 regularization with batch gradient descent to optimize weights.

        Args:
            train (str): Path to clean pickled DataFrame with training examples.
            validation (str): Path to clean pickled DataFrame with validation examples.
            test (str): Path to clean pickled DataFrame with testing examples.
            target (str): A string with column name for response feature.
            rate (float): The learning rate for gradient descent.
            lam (float): The regularization parameter.
            eps (float): The convergence criterion for gradient descent.
            normalize (bool): Indicator for whether or not data should be normalized, default is True.
        """
        with open(train, 'rb') as train_pkl:
            self.train = pickle.load(train_pkl)
        train_cols = self.train.drop(target, axis=1).columns.to_list()

        with open(validation, 'rb') as validation_pkl:
            self.validation = pickle.load(validation_pkl)
        validation_cols = self.validation.drop(target, axis=1).columns.to_list()

        with open(test, 'rb') as test_pkl:
            self.test = pickle.load(test_pkl)
        test_cols = self.test.columns.to_list()

        # Check that columns for train, validation, and test in same order.
        assert np.array_equal(train_cols, validation_cols), 'Train and validation columns not in same order.'
        assert np.array_equal(train_cols, test_cols), 'Train and test columns not in same order.'
        assert np.array_equal(validation_cols, test_cols), 'Validation and test columns not in same order.'

        self.target = target
        self.rate = rate
        self.lam = lam
        self.eps = eps
        self.normalize = normalize

        # Extract features and targets from train, validation, and test sets.
        self.train_features = self.train.drop(target, axis=1)
        self.train_targets = self.train[target]

        self.validation_features = self.validation.drop(target, axis=1)
        self.validation_targets = self.validation[target]

        self.test_features = self.test

        if self.normalize is True:
            # Extract min, max, and range for training features.
            self.train_feature_max = self.train_features.max()
            self.train_feature_min = self.train_features.min()
            self.train_feature_range = self.train_feature_max - self.train_feature_min

            # Normalize all features wrt to training set.
            self.train_features = (self.train_features - self.train_feature_min) / self.train_feature_range
            self.validation_features = (self.validation_features - self.train_feature_min) / self.train_feature_range
            self.test_features = (self.test_features - self.train_feature_min) / self.train_feature_range

            # Extract min, max, and range for training targets.
            self.train_target_max = self.train_targets.max()
            self.train_target_min = self.train_targets.min()
            self.train_target_range = self.train_target_max - self.train_target_min

            # Normalize all targets wrt to training targets.
            self.train_targets = (self.train_targets - self.train_target_min) / self.train_target_range
            self.validation_targets = (self.validation_targets - self.train_target_min) / self.train_target_range

            # Reset dummy variable for intercept to 1.
            self.train_features = self.train_features.assign(dummy=1).set_index('dummy').reset_index()
            self.validation_features = self.validation_features.assign(dummy=1).set_index('dummy').reset_index()
            self.test_features = self.test_features.assign(dummy=1).set_index('dummy').reset_index()

            # Save to .csv to ensure normalization runs correctly.
            my_path = pathlib.Path(__file__).parent.joinpath(pathlib.Path('../data')).resolve()
            out_path = pathlib.Path(__file__).parent.resolve().joinpath(my_path)

            self.train_features.to_csv(out_path.joinpath(pathlib.Path('norm_train_x.csv')))
            self.validation_features.to_csv(out_path.joinpath(pathlib.Path('norm_dev_x.csv')))
            self.test_features.to_csv(out_path.joinpath(pathlib.Path('norm_test_x.csv')))

            self.train_targets.to_csv(out_path.joinpath(pathlib.Path('norm_train_y.csv')), header=self.target)
            self.validation_targets.to_csv(out_path.joinpath(pathlib.Path('norm_dev_y.csv')), header=self.target)

        self.weight_labels = self.train.columns.to_list()

    def train_model(self, max_iter):
        """Trains model with batch gradient descent to optimize weights.

        Calls calc_sse(), calc_gradient(), gradient_descent() from models.gradient_descent.py.

        Args:
            max_iter (int): Maximum number of iterations before terminating training.

        Returns:
            results (dict): Dictionary with lam (regularization parameter), learn_rate (learning rate),
            epsilon  (epsilon for convergence), iterations (number of iterations), convergence (T or F), exploding (
            if gradient
            explodes), labeled_weights (dict with {labels : weights}), weights (nd array of optimized weights),
            train_sse, validation_sse (list with SSE on each set for each iteration), and gradient_norm (list with
            norm of gradient after each iteration).
        """
        # Training and validation sets and labels
        x_train = self.train_features.to_numpy(dtype=np.float64)
        y_train = self.train_targets.to_numpy(dtype=np.float64)
        x_val = self.validation_features.to_numpy(dtype=np.float64)
        y_val = self.validation_targets.to_numpy(dtype=np.float64)

        rate = self.rate
        lam = self.lam
        eps = self.eps
        weight_labels = self.train.drop(self.target, axis=1).columns.tolist()

        print('Initializing training...')

        # Initialize all weights as zero.
        weights = np.zeros(np.size(x_train, axis=1), dtype=np.float64)

        print('Learning rate = ' + str(rate) + ', penalty = ' + str(lam) + ', epsilon = ' + str(eps) + '.')

        # Shape of x[i] is (22, ), shape of y[i] is (1), shape of weights * x[i] - y[i] is (1), shape of grad is (22, ).
        print('Optimizing weights...')

        # Included a progress bar for iteration progress, list for SSE's, indicators for gradient and convergence.
        bar = progressbar.ProgressBar()
        iter_count = 0

        train_sse = []
        val_sse = []
        norm_list = []

        exploding_grad = False
        converge = False

        # Perform batch gradient descent to optimize weights.
        for iteration in bar(range(max_iter)):
            # Calculate gradient and update weights
            gradient = compute_gradient(x_train, y_train, weights)
            weights = gradient_descent(gradient, weights, rate, lam)

            # Calculate sum of squared error for each iteration to store in list.
            train_sse.append(compute_sse(x_train, y_train, weights, lam))
            val_sse.append(compute_sse(x_val, y_val, weights, lam))

            # Calculate norm of gradient to monitor for convergence.
            grad_norm = np.sqrt(gradient.dot(gradient))
            norm_list.append(grad_norm)

            # Check for divergence with the norm of the gradient to see if exploding.
            if np.isinf(grad_norm):
                print('\nGradient exploding.\n')
                exploding_grad = True
                break

            # Check for convergence using the norm of the gradient.
            if grad_norm <= eps:
                print('\nConvergence achieved with epsilon = ' + str(eps) + ' in ' + str(iteration) + ' iterations.')
                converge = True
                break

            # Check that gradient is still decreasing sufficiently
            if iter_count > 10000 and abs(norm_list[-2] - norm_list[-1]) < 1e-06:
                print('\nGradient not decreasing significantly.')
                converge = True
                break

        # If we haven't converged by this point might as well stop and figure out why.
        if iter_count == max_iter:
            print('Maximum iterations reached without convergence.\n')
        weights = weights.tolist()
        labeled_weights = dict(zip(weight_labels, weights))

        results = {'lam': lam,
                   'learn_rate': rate,
                   'epsilon': eps,
                   'iterations': iter_count,
                   'convergence': converge,
                   'exploding': exploding_grad,
                   'labeled_weights': labeled_weights,
                   'weights': weights,
                   'train_sse': train_sse,
                   'validation_sse': val_sse,
                   'gradient_norm': norm_list}
        return results

    def predict_validation(self, weights):
        """Generates predictions and sum-of-squared error calculations for validation data with labels.

        Uses validation set provided when instance of class is created.

        Args:
            weights (ndarray): (, n) ndarray of weights produced from training.

        Returns:
            results (dict): Dictionary with lam (regularization parameter), predictions (list of predictions),
            and SSE (SSE for predictions).
        """
        lam = self.lam
        x_val = self.validation_features.to_numpy(dtype=np.float64)
        y_val = self.validation_targets.to_numpy(dtype=np.float64)
        predictions = predict(x_val, weights)
        sse = compute_sse(x_val, y_val, weights, lam)
        results = {'lam': lam,
                   'SSE': sse,
                   'predictions': predictions}
        return results

    def predict_test(self, weights):
        """Generates predictions for unlabeled test data, transformed back to original scale if necessary.

        Args:
            weights (ndarray): (, n) ndarray of weights produced from training.

        Returns:
            results (dict): Dictionary with lam (regularization parameter) and predictions (list of predictions).
        """
        lam = self.lam
        x_test = self.test_features.to_numpy(dtype=np.float64)
        predictions = predict(x_test, weights)
        if self.normalize is True:
            target_max = self.train_target_max
            target_min = self.train_target_min
            predictions = [pred * (target_max - target_min) + target_min for pred in predictions]
        results = {'lam': lam,
                   'predictions': predictions}
        return results
