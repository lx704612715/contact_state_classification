from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import contact_state_classification as csc



def main():
    experiment_dir = csc.config.path["experiment_dir"]
    cs_classifier = csc.CSClassifier(experiment_dir=experiment_dir,
                                     dataset_name_list=csc.config.path["dataset"],
                                     test_set_name_list=csc.config.path["test_set"])


if __name__ == "__main__":
    main()