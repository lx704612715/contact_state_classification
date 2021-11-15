# This is where all file names and path-related parameters are stored.
path = {
    "experiment_dir": "contact_state_classification/tests/1908_hfv",
    "dataset_name": "RoboticsProject2510_2"
}
# This is where all classifier configuration parameters are stored.
# Since different classifiers may be used, parameters may need to be nested.
params = {
    "n_act": 12,
    "simple_features": ["dist"],
    "complex_features": [],
    "n_splits": 8,
    "n_neighbors": 3,
    "classifier": "KNN"
}
