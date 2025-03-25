import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

script_dir = os.path.dirname(__file__)

datasets = {
    "c300": {
        "100": {
            "train": f"{script_dir}/Datasets/train_c300_d100.csv",
            "validation": f"{script_dir}/Datasets/validation_c300_d100.csv",
            "test": f"{script_dir}/Datasets/test_c300_d100.csv",
        },
        "1000": {
            "train": f"{script_dir}/Datasets/train_c300_d1000.csv",
            "validation": f"{script_dir}/Datasets/validation_c300_d1000.csv",
            "test": f"{script_dir}/Datasets/test_c300_d1000.csv",
        },
        "5000": {
            "train": f"{script_dir}/Datasets/train_c300_d5000.csv",
            "validation": f"{script_dir}/Datasets/validation_c300_d5000.csv",
            "test": f"{script_dir}/Datasets/test_c300_d5000.csv",
        },
    },
    "c500": {
        "100": {
            "train": f"{script_dir}/Datasets/train_c500_d100.csv",
            "validation": f"{script_dir}/Datasets/validation_c500_d100.csv",
            "test": f"{script_dir}/Datasets/test_c500_d100.csv",
        },
        "1000": {
            "train": f"{script_dir}/Datasets/train_c500_d1000.csv",
            "validation": f"{script_dir}/Datasets/validation_c500_d1000.csv",
            "test": f"{script_dir}/Datasets/test_c500_d1000.csv",
        },
        "5000": {
            "train": f"{script_dir}/Datasets/train_c500_d5000.csv",
            "validation": f"{script_dir}/Datasets/validation_c500_d5000.csv",
            "test": f"{script_dir}/Datasets/test_c500_d5000.csv",
        },
    },
    "c1000":{
        "100": {
            "train": f"{script_dir}/Datasets/train_c1000_d100.csv",
            "validation": f"{script_dir}/Datasets/validation_c1000_d100.csv",
            "test": f"{script_dir}/Datasets/test_c1000_d100.csv",
        },
        "1000": {
            "train": f"{script_dir}/Datasets/train_c1000_d1000.csv",
            "validation": f"{script_dir}/Datasets/validation_c1000_d1000.csv",
            "test": f"{script_dir}/Datasets/test_c1000_d1000.csv",
        },
        "5000": {
            "train": f"{script_dir}/Datasets/train_c1000_d5000.csv",
            "validation": f"{script_dir}/Datasets/validation_c1000_d5000.csv",
            "test": f"{script_dir}/Datasets/test_c1000_d5000.csv",
        },
    },
    "c1500":{
        "100": {
            "train": f"{script_dir}/Datasets/train_c1500_d100.csv",
            "validation": f"{script_dir}/Datasets/validation_c1500_d100.csv",
            "test": f"{script_dir}/Datasets/test_c1500_d100.csv",
        },
        "1000": {
            "train": f"{script_dir}/Datasets/train_c1500_d1000.csv",
            "validation": f"{script_dir}/Datasets/validation_c1500_d1000.csv",
            "test": f"{script_dir}/Datasets/test_c1500_d1000.csv",
        },
        "5000": {
            "train": f"{script_dir}/Datasets/train_c1500_d5000.csv",
            "validation": f"{script_dir}/Datasets/validation_c1500_d5000.csv",
            "test": f"{script_dir}/Datasets/test_c1500_d5000.csv",
        },
    },
    "c1800":{
        "100": {
            "train": f"{script_dir}/Datasets/train_c1800_d100.csv",
            "validation": f"{script_dir}/Datasets/validation_c1800_d100.csv",
            "test": f"{script_dir}/Datasets/test_c1800_d100.csv",
        },
        "1000": {
            "train": f"{script_dir}/Datasets/train_c1800_d1000.csv",
            "validation": f"{script_dir}/Datasets/validation_c1800_d1000.csv",
            "test": f"{script_dir}/Datasets/test_c1800_d1000.csv",
        },
        "5000": {
            "train": f"{script_dir}/Datasets/train_c1800_d5000.csv",
            "validation": f"{script_dir}/Datasets/validation_c1800_d5000.csv",
            "test": f"{script_dir}/Datasets/test_c1800_d5000.csv",
        },
    }
}

def load_dataset(file_path):
    return pd.read_csv(file_path)

def train_and_evaluate(dataset_paths):
    train_data = load_dataset(dataset_paths["train"])
    validation_data = load_dataset(dataset_paths["validation"])
    test_data = load_dataset(dataset_paths["test"])

    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_val = validation_data.drop('label', axis=1)
    y_val = validation_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 40, 50]
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    # Combine training and validation sets
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    best_model = DecisionTreeClassifier(**best_params)
    best_model.fit(X_combined, y_combined)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return best_params, accuracy, f1

results = {}
for category, sizes in datasets.items():
    results[category] = {}
    for size, paths in sizes.items():
        best_params, accuracy, f1 = train_and_evaluate(paths)
        results[category][size] = {
            'best_params': best_params,
            'accuracy': accuracy,
            'f1_score': f1
        }

for category, sizes in results.items():
    for size, metrics in sizes.items():
        print(f"Category: {category}, Size: {size}")
        print(f"Best Params: {metrics['best_params']}")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"F1 Score: {metrics['f1_score']}")
        print()

