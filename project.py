import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import fetch_openml

script_dir = os.path.dirname(__file__)  # Absolute directory path to the file dir on the users system

# Setting up the relative filepaths for the datasets
datasets = {
    "c300": {
        "100": {
            "train": f"{script_dir}/Datasets/train_c300_d100.csv",
            "valid": f"{script_dir}/Datasets/valid_c300_d100.csv",
            "test": f"{script_dir}/Datasets/test_c300_d100.csv",
        },
        "1000": {
            "train": f"{script_dir}/Datasets/train_c300_d1000.csv",
            "valid": f"{script_dir}/Datasets/valid_c300_d1000.csv",
            "test": f"{script_dir}/Datasets/test_c300_d1000.csv",
        },
        "5000": {
            "train": f"{script_dir}/Datasets/train_c300_d5000.csv",
            "valid": f"{script_dir}/Datasets/valid_c300_d5000.csv",
            "test": f"{script_dir}/Datasets/test_c300_d5000.csv",
        },
    },
    # ... (rest of the datasets dictionary remains the same)
}

def load_dataset(file_path):
    """
    Load dataset with error handling for column names
    """
    try:
        df = pd.read_csv(file_path)
        
        # Find the target column (label)
        label_column = None
        for col in df.columns:
            if col.lower() in ['label', 'target', 'class', 'y']:
                label_column = col
                break
        
        if label_column is None:
            raise ValueError(f"No label column found in {file_path}")
        
        return df, label_column
    except Exception as e:
        print(f"Error loading dataset {file_path}: {e}")
        return None, None

def train_and_evaluate_classifiers(dataset_paths):
    """
    Perform experiments for different classifiers on a given dataset
    """
    # Load datasets
    train_data, train_label_col = load_dataset(dataset_paths["train"])
    valid_data, valid_label_col = load_dataset(dataset_paths["valid"])
    test_data, test_label_col = load_dataset(dataset_paths["test"])

    if train_data is None or valid_data is None or test_data is None:
        print("Failed to load one or more datasets")
        return None

    # Prepare data
    X_train = train_data.drop(train_label_col, axis=1)
    y_train = train_data[train_label_col]
    X_val = valid_data.drop(valid_label_col, axis=1)
    y_val = valid_data[valid_label_col]
    X_test = test_data.drop(test_label_col, axis=1)
    y_test = test_data[test_label_col]

    # Rest of the function remains the same as in the original script
    # ... (classifiers, grid search, etc.)

    # Classifiers and their parameter grids
    classifiers = {
        'DecisionTree': {
            'classifier': DecisionTreeClassifier(),
            'param_grid': {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 10, 20, 30, 40, 50]
            }
        },
        'Bagging': {
            'classifier': BaggingClassifier(base_estimator=DecisionTreeClassifier()),
            'param_grid': {
                'n_estimators': [10, 50, 100],
                'max_samples': [0.5, 0.7, 1.0],
                'max_features': [0.5, 0.7, 1.0]
            }
        },
        'RandomForest': {
            'classifier': RandomForestClassifier(),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30]
            }
        },
        'GradientBoosting': {
            'classifier': GradientBoostingClassifier(),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5],
                'max_depth': [3, 4, 5]
            }
        }
    }

    # Results dictionary to store experiment results
    results = {}

    # Perform experiments for each classifier
    for name, config in classifiers.items():
        # Perform grid search
        grid_search = GridSearchCV(config['classifier'], config['param_grid'], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Get best parameters
        best_params = grid_search.best_params_

        # Combine training and validation sets
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])

        # Train best model
        best_model = type(config['classifier'])(**best_params)
        best_model.fit(X_combined, y_combined)

        # Predict and evaluate
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Store results
        results[name] = {
            'best_params': best_params,
            'accuracy': accuracy,
            'f1_score': f1
        }

    return results

def mnist_experiment():
    """
    Perform MNIST dataset experiment
    """
    # Load MNIST dataset
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X / 255.0  # Normalize pixel values to [0,1]
    
    # Split into training (60K) and test (10K) sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Classifiers for MNIST
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(),
        'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier()),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }

    # MNIST results
    mnist_results = {}

    # Perform MNIST experiment
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        mnist_results[name] = accuracy

    return mnist_results

def main():
    # Experiment results for all datasets
    all_results = {}

    # Perform experiments for each dataset category and size
    for category, sizes in datasets.items():
        all_results[category] = {}
        for size, paths in sizes.items():
            print(f"\nExperiments for Category: {category}, Size: {size}")
            results = train_and_evaluate_classifiers(paths)
            
            if results is not None:
                all_results[category][size] = results

                # Print results for this dataset
                for clf_name, metrics in results.items():
                    print(f"\n{clf_name} Results:")
                    print(f"Best Params: {metrics['best_params']}")
                    print(f"Accuracy: {metrics['accuracy']}")
                    print(f"F1 Score: {metrics['f1_score']}")

    # MNIST Experiment
    print("\nMNIST Dataset Experiment:")
    mnist_results = mnist_experiment()
    for name, accuracy in mnist_results.items():
        print(f"{name} - Accuracy: {accuracy}")

    return all_results, mnist_results

if __name__ == "__main__":
    final_results, mnist_results = main()