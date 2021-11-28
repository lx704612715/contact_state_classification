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
    "basic_visualization": True,
    "simple_features": ["dist", "d_ee_theta", "d_ee_phi", "obs_ee_theta", "obs_ee_phi"],
    "complex_features": ["error_q", "base_s_ee"],
    "n_splits": 8,
    "n_neighbors": 4,
    "n_components": 3,
    "classifier": "SHP",
    "cs_index_map": {"CS1": 1, "CS2": 2, "CS3": 3, "CS5": 4, "CS6": 5}
}
