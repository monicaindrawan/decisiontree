import numpy as np
from tree import DecisionTree
from evaluation import *

# For reproducibility
random_generator = np.random.default_rng(seed = 42)

def cross_validation(X, y, k=10):
    """
    Evalute the decision tree over k folds,

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target matrix.
        k (int) : number of folds training data is split into

    Returns:
        average_accuracy : int
            The average accuracy with regards to the validation data
    """

    confusion_matrix_list = []
    count = 0

    # Splits data into training and test folds, iterates k times with each fold as test fold once
    for train_indices, test_indices in train_test_k_fold(k, len(X), random_generator):
        x_train = X[train_indices, :]
        y_train = y[train_indices]
        x_test = X[test_indices, :]
        y_test = y[test_indices]

        # Create, fit, and predict with decision tree
        tree = DecisionTree()
        tree.fit(x_train, y_train)
        predictions = tree.predict(x_test)
        confusion_matrix_list.append(confusion_matrix(y_test, predictions))

        count += 1
        print("Iteration:", f"{count}/{k}", end='\r')

    # Evaluation calculations over all CV runs
    average_confusion_matrix = np.sum(confusion_matrix_list, axis=0)/(k)
    average_accuracy = accuracy_from_confusion(average_confusion_matrix)
    average_precision = precision_from_confusion(average_confusion_matrix)
    average_recall = recall_from_confusion(average_confusion_matrix)
    average_f1 = f1_score(average_precision[0], average_recall[0])

    return (average_accuracy, average_precision[1], average_recall[1], average_f1[1])



def nested_cross_validation(X, y, n_outer_folds = 10, n_inner_folds = 9):
    """
    Evalute the pruned decision tree over n_outer_fold test sets with n_inner_folds validation sets per test set for a total of 'n_outer_fold*n_inner_fold trees'.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target matrix.
        n_outer_folds (int) : Number of folds that data is split into (trainval and train folds)
        n_inner_folds (int) : Number of folds that trainval is split into (train and validation folds)

    Returns:
        tuple: A tuple containing the averaged performance metrics:
            (average_accuracy, average_precision, average_recall, average_f1,
            average_max_depth_before, average_max_depth_after)
            where:
                - average_accuracy (float): Mean accuracy across all test folds.
                - average_precision (float): Mean precision across all test folds (assumes binary classification, returning the metric for the positive class [1]).
                - average_recall (float): Mean recall across all test folds (assumes binary classification, returning the metric for the positive class [1]).
                - average_f1 (float): Mean F1-score across all test folds (assumes binary classification, returning the metric for the positive class [1]).
                - average_max_depth_before (float): Mean maximum depth of the tree before pruning.
                - average_max_depth_after (float): Mean maximum depth of the tree after pruning on the validation set.
    """
    count = 0
    confusion_matrix_list = []
    total_max_depth_before = 0
    total_max_depth_after = 0

    # Splits data into trainval and test folds, iterates k times with each fold as test fold once
    for i, (trainval_indices, test_indices) in enumerate(train_test_k_fold(n_outer_folds, len(X), random_generator)):
        x_trainval = X[trainval_indices, :]
        y_trainval = y[trainval_indices]
        x_test = X[test_indices, :]
        y_test = y[test_indices]

        # Splits trainval fold into seperate training and validation folds
        splits = train_test_k_fold(n_inner_folds, len(x_trainval), random_generator)

        # Loops so each fold from trainval is validation fold once
        for (train_indices, val_indices) in splits:
            x_train = x_trainval[train_indices, :]
            y_train = y_trainval[train_indices]
            x_val = x_trainval[val_indices, :]
            y_val = y_trainval[val_indices]

            # Creating decicion tree insance then training, pruning, and predicting
            tree = DecisionTree()
            tree.fit(x_train, y_train)

            max_depth_before, max_depth_after = tree.prune_tree(x_val, y_val)
            total_max_depth_before += max_depth_before
            total_max_depth_after += max_depth_after
            
            predictions = tree.predict(x_test)
            confusion_matrix_list.append(confusion_matrix(y_test, predictions))
            count += 1
            print("Iteration:", f"{count}/{n_outer_folds * n_inner_folds}", end='\r')

    # Evaluation calculations over all CV runs
    average_confusion_matrix = np.sum(confusion_matrix_list, axis=0)/(n_outer_folds*n_inner_folds)
    average_accuracy = accuracy_from_confusion(average_confusion_matrix)
    average_precision = precision_from_confusion(average_confusion_matrix)
    average_recall = recall_from_confusion(average_confusion_matrix)
    average_f1 = f1_score(average_precision[0], average_recall[0])

    average_max_depth_before = total_max_depth_before / count
    average_max_depth_after = total_max_depth_after / count

    return (average_accuracy, average_precision[1], average_recall[1], average_f1[1], average_max_depth_before, average_max_depth_after)
