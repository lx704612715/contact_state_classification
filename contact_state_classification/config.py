# This is where all file names and path-related parameters are stored.
path = {
    "experiment_dir": "contact_state_classification/tests/1908_hfv",
    "dataset_name": "RoboticsProject2510_2"
}
# This is where all classifier configuration parameters are stored.
# Since different classifiers may be used, parameters may need to be nested.
params = {
    "simple_features": ["dist", "obs_ee_theta", "obs_ee_phi"],
    "complex_features": ["error_q"],
    "classifier": "SOM"
}
