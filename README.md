# Decision Tree Group Project

A Python implementation of a decision tree classifier from scratch. The classifier trained on datasets to determine one of four possible locations based on the signal strength data collected from a mobile phone. 


## Core Features

- Decision tree constructed from maximising information gain calculated using entropy to find optimal splits.
- Tree pruning to reduce overfitting by replacing 2 child nodes with a single node if it improves or maintains accuracy.
- Tree visualisation.
- Evaluation metrics and confusion matrix using accuracy, precision, recall and F1-score.
- Standard and nested cross validation implementations for unpruned and pruned decision trees.


## Project Structure

All Python source files are under the root directory
- `main.py`: The main and the entry point of the project. The final and current iteration of the file performs the standard and the nested cross validation for both the clean and noisy datasets, and prints out the evaluation metrics.
- `tree.py`: Contains the core `DecisionTree` class, including the methods for `fit`, `predict` and `prune_tree`.
- `support.py`: Includes the `load_data` function as well as the `find_best_split_feature` function for information gain calculations used to construct the tree.
- `evaluation.py`: Contains functions for obtaining the confusion matrix and evaluation metrics.
- `cross_eval.py`: Implements both standard `cross_validation` and `nested_cross_validation` functions.
- `visualisation.py`: Implements the `plot_tree` function to visualise the decision tree.
- `wifi_db`: Folder which contains the clean and noisy datasets.


## Getting Started

### Prerequisites

Besides the standard Python libraries, the project uses `numpy`, make sure that it is available in the active Python environment. Optionally, to run the tree visualisation code in visualisation.py, `matplotlib` is also required.


### To Run the Project

1. Make sure to change to the root directory of the project, which is the directory containing all of the Python source files.
2. Run the main Python file by executing the command `python -u main.py`.


The final version of the `main.py` abstracts most of the functionalities into 2 variants of the cross validation functions, where these function calls handle all of tree construction, fitting and evaluation, with the goal of producing the final cross validated metrics. As such, the individual component calls are not present in `main.py`. Furthermore, the lab machines' Python environment do not contain `matplotlib`, therefore `visualisation.py` cannot be tested on them. An example is provided below to demonstrate these components not shown in `main.py`:

## Example Code
### Tree construction and visualisation:
````python
from support import load_data, train_test_split
from tree import DecisionTree
# IMPORTANT: Make sure matplotlib is available, if not, do not includes this module
from visualisation import plot_tree
from evaluation import *


if __name__ == "__main__":
    # Load the data
    X, y = load_data("wifi_db/noisy_dataset.txt")     # Replace load_data argument with the path of any data.txt files

    # Split the dataset for training and testing
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y)

    # Construct and fit the decision tree classifier
    decision_tree_classifier = DecisionTree()
    decision_tree_classifier.fit(X_train, y_train)
    
    # Prune the tree
    decision_tree_classifier.prune_tree(X_val, y_val)

    # Predict the test dataset
    y_prediction = decision_tree_classifier.predict(X_test)

    # Show the Confusion Matrix
    print(confusion_matrix(y_test, y_prediction))

    # IMPORTANT: Make sure matplotlib is available
    # Visualise the pruned tree
    plot_tree(decision_tree_classifier)
````