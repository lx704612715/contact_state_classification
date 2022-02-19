# This is where all file names and path-related parameters are stored.
path = {
    "experiment_dir": "contact_state_classification/tests/1908_hfv",
    "dataset": ["RoboticsProject0301"],
    "test_set": ["RoboticsProject0301"]
}
# This is where all classifier configuration parameters are stored.
# Since different classifiers may be used, parameters may need to be nested.
params = {
    "n_act": 12,
    "use_pca": False,
    "basic_visualization": False,
    "circular_splicing": False,
    "interpolation_method": "linear",
    "upsampling_rate": 16,
    "shaplet_size": 0.5,
    "simple_features": ["dist"],
    "complex_features": [],
    "n_splits": 8,
    "n_neighbors": 4,
    "n_components": 3,
    "classifier": "KNN",
    "cs_index_map": {"CS1": 1, "CS2": 2, "CS3": 3, "CS5": 4, "CS6": 5}
}

N_ACT = 12
USE_PCA = False
BASIC_VISUALIZATION = True
CIRCULAR_SPLICING = False
INTERPOLATION_METHOD = "linear"  # [linear, quadratic, cubic]
UPSAMPLING_RATE = 16
SHAPELET_SIZE = 0.5
SIMPLE_FEATURES = ["dist"]
COMPLEX_FEATURES = []
N_SPLITS = 8
N_NEIGHBORS = 4
N_COMPONENTS = 3
CLASSIFIER = "SHP"  # [KNN, SHP]
CS_INDEX_MAP = {"CS1": 1, "CS2": 2, "CS3": 3, "CS4": 4, "CS5": 5, "CS6": 6}
