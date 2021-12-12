# This is where all file names and path-related parameters are stored.
path = {
    "experiment_dir": "contact_state_classification/tests/1908_hfv",
    "dataset": ["RoboticsProject0112", "RoboticsProject2510"],
    "test_set": ["RoboticsProject2510"]
}
# This is where all classifier configuration parameters are stored.
# Since different classifiers may be used, parameters may need to be nested.
params = {
    "n_act": 12,
    "use_pca": False,
    "basic_visualization": False,
    "circular_splicing": False,
    "interpolation_method": "quadratic",
    "upsampling_rate": 16,
    "shaplet_size": 0.5,
    "simple_features": ["dist", "d_ee_theta", "d_ee_phi", "obs_ee_theta", "obs_ee_phi"],
    "complex_features": ["error_q"],
    "n_splits": 8,
    "n_neighbors": 4,
    "n_components": 3,
    "classifier": "SHP",
    "cs_index_map": {"CS1": 1, "CS2": 2, "CS3": 3, "CS5": 4, "CS6": 5}
}

N_ACT = 12
N_SPLITS = 8