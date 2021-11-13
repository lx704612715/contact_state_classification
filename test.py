import pandas as pd
import os
import numpy as np
from loguru import logger
import sys
# Load dateloader
from scipy.stats import gaussian_kde
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import contact_state_classification as csc
import numpy as np
from contact_state_classification import config as cfg
import pandas as pd
import seaborn as sns


def main():
    experiment_dir = csc.config.path["experiment_dir"]
    cs_classifier = csc.CSClassifier(experiment_dir=experiment_dir, dataset_name=csc.config.path["dataset_name"])
    cs_classifier.pca()
    test_idx = [74]
    df = cs_classifier.csd_data_df.iloc[test_idx]
    
    X, y = csc.CSClassifier.extract_features_from_df(df)
    print(cs_classifier.predict(input_data=X))
    print(y)

if __name__ == "__main__":
    main()