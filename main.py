from evaluation import *
from cross_eval import cross_validation, nested_cross_validation
from support import load_data


if __name__ == "__main__":
    X_noisy, y_noisy = load_data("wifi_db/noisy_dataset.txt")
    X_clean, y_clean = load_data("wifi_db/clean_dataset.txt")

    # Run this for cross validation on unpruned tree
    print("Performing cross validation on the clean dataset")
    cv_clean_metrics = cross_validation(X_clean, y_clean)
    print("Performing cross validation on the noisy dataset")
    cv_noisy_metrics = cross_validation(X_noisy, y_noisy)

    # Run this for nested cross validation on pruned tree
    print("Performing nested cross validation on the clean dataset")
    cv_nested_clean_metrics = nested_cross_validation(X_clean, y_clean)
    print("Performing nested cross validation on the noisy dataset")
    cv_nested_noisy_metrics = nested_cross_validation(X_noisy, y_noisy)

    # Print out metrics for all 4 cross validations
    print('\n')
    print("Cross validation metrics for unpruned tree on the clean dataset")
    print("  Accuracy: ", cv_clean_metrics[0], '\n',
           " Precision: ", cv_clean_metrics[1], '\n', 
           " Recall: ", cv_clean_metrics[2], '\n', 
           " F1: ", cv_clean_metrics[3], '\n')
    print("Cross validation metrics for unpruned tree on the noisy dataset")
    print("  Accuracy: ", cv_noisy_metrics[0], '\n', 
          " Precision: ", cv_noisy_metrics[1], '\n', 
          " Recall: ", cv_noisy_metrics[2], '\n', 
          " F1: ", cv_noisy_metrics[3], '\n')
    print("Cross validation metrics for pruned tree on the clean dataset")
    print("  Accuracy: ", cv_nested_clean_metrics[0], '\n', 
          " Precision: ", cv_nested_clean_metrics[1], '\n', 
          " Recall: ", cv_nested_clean_metrics[2], '\n', 
          " F1: ", cv_nested_clean_metrics[3], '\n', 
          " Average maximum depth before pruning: ", cv_nested_clean_metrics[4], '\n', 
          " Average maximum depth after pruning: ", cv_nested_clean_metrics[5], '\n')
    print("Cross validation metrics for pruned tree on the noisy dataset")
    print("  Accuracy: ", cv_nested_noisy_metrics[0], '\n', 
          " Precision: ", cv_nested_noisy_metrics[1], '\n', 
          " Recall: ", cv_nested_noisy_metrics[2], '\n', 
          " F1: ", cv_nested_noisy_metrics[3], '\n', 
          " Average maximum depth before pruning: ", cv_nested_noisy_metrics[4], '\n', 
          " Average maximum depth after pruning: ", cv_nested_noisy_metrics[5])