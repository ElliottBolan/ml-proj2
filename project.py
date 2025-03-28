# Elliott Bolan, EXB210027, CS 4375.004, Start Date: 3/23/2025
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import fetch_openml
from joblib import Parallel, delayed

script_dir = os.path.dirname(__file__)

# Setting up the relative filepaths for the datasets
datasets = {
    f'c{clauses}': {
        str(examples): {
            dataset_type: f"{script_dir}/Datasets/{dataset_type}_c{clauses}_d{examples}.csv"
            for dataset_type in ['train', 'valid', 'test']
        } for examples in ['100', '1000', '5000']
    } for clauses in ['300', '500', '1000', '1500', '1800']
}

def load_dataset(file_path): # function to load a dataset from a given file path
    try:
        df = pd.read_csv(file_path, header=None)
        label_column = df.columns[-1]
        return df, label_column
    except Exception as e:
        print(f"Error loading dataset {file_path}: {e}")
        return None, None

def train_and_evaluate_classifier(name, config, X_train, y_train, X_val, y_val, X_test, y_test):
    # Perform grid search
    grid_search = GridSearchCV(config['classifier'], config['param_grid'], cv=5, scoring='f1_weighted', n_jobs=-1)
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

    return name, {
        'best_params': best_params,
        'accuracy': accuracy,
        'f1_score': f1
    }

def train_and_evaluate_classifiers(dataset_paths):
    train_data, train_label_col = load_dataset(dataset_paths["train"])
    valid_data, valid_label_col = load_dataset(dataset_paths["valid"])
    test_data, test_label_col = load_dataset(dataset_paths["test"])

    # Error handling for loading datasets
    if train_data is None or valid_data is None or test_data is None:
        print("Failed to load one or more datasets")
        return None

    # Ensure the label columns are consistent across datasets
    X_train = train_data.drop(train_label_col, axis=1)
    y_train = train_data[train_label_col]
    X_val = valid_data.drop(valid_label_col, axis=1)
    y_val = valid_data[valid_label_col]
    X_test = test_data.drop(test_label_col, axis=1)
    y_test = test_data[test_label_col]

    # Dictionary of classifiers and their parameter grids for hyperparameter tuning
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
            'classifier': BaggingClassifier(estimator=DecisionTreeClassifier()),
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

    # Used parallel processing to speed up the process, sorry to the TAs for the extra load on the system!
    results = Parallel(n_jobs=-1)(
        delayed(train_and_evaluate_classifier)(name, config, X_train, y_train, X_val, y_val, X_test, y_test)
        for name, config in classifiers.items()
    )

    return dict(results)

def mnist_experiment():
    # Downloading and preprocessing the MNIST dataset 
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False) # as_frame=True would return the dataset as a pandas DataFrame
    X = X / 255.0
    # Split into training (60K) and test (10K) sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Dictionary for classifiers to be used for the MNIST experiment
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(),
        'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier()),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }

    # Fitting each classifier on the MNIST dataset and evaluating their performance
    mnist_results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        mnist_results[name] = accuracy

    return mnist_results

def main():
    all_results = {}
    for category, sizes in datasets.items():
        all_results[category] = {}
        for size, paths in sizes.items():
            print(f"\nExperiments for Category: {category}, Size: {size}")
            results = train_and_evaluate_classifiers(paths)
            if results is not None:
                all_results[category][size] = results
                # Printing results
                for clf_name, metrics in results.items():
                    print(f"\n{clf_name} Results:")
                    print(f"Best Params: {metrics['best_params']}")
                    print(f"Accuracy: {metrics['accuracy']}")
                    print(f"F1 Score: {metrics['f1_score']}")

    # Running the MINST experiement and printing the results
    print("\nMNIST Dataset Experiment:")
    mnist_results = mnist_experiment()
    for name, accuracy in mnist_results.items():
        print(f"{name} - Accuracy: {accuracy}")

    return all_results, mnist_results

if __name__ == "__main__":
    final_results, mnist_results = main()