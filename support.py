import numpy as np
from evaluation import train_test_k_fold


def load_data(file):
    """
    Loads data from a text file.

    Assumes the last column is the target variable (y) and all preceding
    columns are features (x).

    Args:
        file (str): The path to the data file

    Returns:
        x (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector, converted to integer type.
    """
    data = np.loadtxt(file)

    x = data[:, :-1]
    y = data[:, -1].astype(int)

    return x, y


def train_test_split(X, y, k=10):
    """
    Splits the X and y into train, validation and test datasets

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target array of shape (n_samples,) or (n_samples, 1).
        k (int, optional): Number of folds used to generate indices. Default is 10.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.
    """
    fold = train_test_k_fold(k, len(X))[0]
    train_val_indices, test_indices = fold[0], fold[1]
    fold2 = train_test_k_fold(k, len(train_val_indices))[0]
    train_indices, val_indices = fold2[0], fold2[1]


    X_trainval = X[train_val_indices, :]
    y_trainval = y[train_val_indices]
    X_test = X[test_indices, :]
    y_test = y[test_indices]
    X_train = X_trainval[train_indices, :]
    y_train = y_trainval[train_indices]
    X_val = X_trainval[val_indices, :]
    y_val = y_trainval[val_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test 


def entropy(y):
    """
    Calculates the Shannon entropy of a set of labels.

    Args:
        y (np.ndarray):An array of class labels.

    Returns:
        s (float): The calculated Shannon entropy.
    """
    s = 0
    for class_label in np.unique(y):
        p = np.sum(class_label == y) / len(y)
        s -= p * np.log2(p)

    return s


def find_ordered_value_changes(X, y, index):
    """
    Identify positions where each feature column of X changes when the dataset is sorted by it.

    Args:
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix containing the input data.

        y : np.ndarray, shape (n_samples,)
            Target vector or class labels corresponding to each sample in `X`.

        index : int
            The column index of `X` to use for sorting and analysis.

    Returns:
        change_index : list of int
            Indices where the sorted feature values in X[:, index] change 
            as the values increase.  

        X_sorted : np.ndarray, shape (n_samples, n_features)
            The feature matrix sorted by column `index`.

        y_sorted : np.ndarray, shape (n_samples,)
            The target vector sorted according to column `index`.
    """
    
    # locate the feature column
    x = X[:, index]
    # sort based on x feature values
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    x_current = x_sorted[0]
    change_index = []
    # identify the points where x values have changed
    for i in range(1, len(x_sorted)):
        if x_sorted[i] != x_current:
            change_index.append(i)
        x_current = x_sorted[i]
    return change_index, x_sorted, y_sorted



def find_ideal_split(x_sorted, y_sorted, splits):
    """
    Finds the optimal split condition for a feature (x) based on maximizing information gain
    for the target variable (y).

    Args:
        x_sorted (np.array): A list or array of **sorted** feature values.
        y_sorted (np.array): A list or array of corresponding target/class
                             labels, **sorted** according to the order in `x_sorted`.
        splits (list of int): A list of integer indices where a split can occur.
                             These indices indicate the point *after* which the
                             data is split (i.e., the first element of the right child).

    Returns:
        split_value (float): The feature value (split point) that yields the maximum
                             information gain. This is calculated as the midpoint
                             between the last element of the left split and the
                             first element of the right split.
        max_gain (float): The maximum information gain achieved at the best split.
    """
       
    # calculate total entropy
    total_entropy = entropy(y_sorted)

    split_value = 0
    max_gain = 0

    for s in splits:
        # calculate the entropy of each split
        left_list = y_sorted[:s]
        right_list = y_sorted[s:]
        left_entropy = entropy(left_list)
        right_entropy = entropy(right_list)

        # calculate the weighted entropy and the gain
        weighted_entropy = len(left_list) / len(y_sorted) * left_entropy + len(right_list) / len(y_sorted) * right_entropy
        gain = total_entropy - weighted_entropy

        # if the gain is bigger than the previously calculated gain, find the split condition
        if gain > max_gain:
            split_value = (x_sorted[s-1] + x_sorted[s]) / 2
            max_gain = gain

    return split_value, max_gain



def find_best_split_feature(X, y):
    """
    Finds the best feature and split value to divide the dataset.

    Iterates through all features, finds the optimal split point for each,
    and returns the feature and split value that yield the highest
    information gain.

    Args:
        X (np.ndarray): Feature matrix containing the input data.
        y (np.ndarray): Target vector or class labels.

    Returns:
        best_feature_index (int): The column index of the feature that provides the maximum information gain.
        best_split_value (float): The threshold value for the best feature to split on.
    """
    max_gain = 0
    best_split_value = 0
    best_feature_index = 0

    # Loop through all feature columns
    for feature_index in range(X.shape[1]):
        # Find the sorted array and split points
        splits, x_sorted, y_sorted = find_ordered_value_changes(X, y, feature_index)
        # Calculate the best split point on the current feature that yielded highest information gain
        feature_split_value, feature_gain = find_ideal_split(x_sorted, y_sorted, splits)

        if feature_index == 0:
            best_split_value = feature_split_value

        # Get the best feature the split the tree on and its split condition value
        if feature_gain > max_gain:
            best_split_value = feature_split_value
            max_gain = feature_gain
            best_feature_index = feature_index

    return best_feature_index, best_split_value