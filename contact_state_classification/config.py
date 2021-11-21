# This is where all file names and path-related parameters are stored.
path = {
    "experiment_dir": "contact_state_classification/tests/1908_hfv",
    "dataset_name": "RoboticsProject2510_2"
}
# This is where all classifier configuration parameters are stored.
# Since different classifiers may be used, parameters may need to be nested.
params = {
    "n_act": 12,
    "use_pca": True,
    "simple_features": ["dist", "obs_ee_theta", "obs_ee_phi"],
    "complex_features": ["error_q"],
    "n_splits": 8,
    "n_neighbors": 4,
    "n_components": 2,
    "classifier": "KNN"
}
