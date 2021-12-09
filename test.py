from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
from visdom import Visdom
import numpy as np
import pandas as pd
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib
from contact_state_classification import config as cfg

# Load dateloader
import contact_state_classification as csc
from contact_state_classification import utils
from scipy.interpolate import interp1d


def main():
    experiment_dir = csc.config.path["experiment_dir"]
    cs_classifier = csc.CSClassifier(experiment_dir=experiment_dir,
                                     dataset_name_list=csc.config.path["dataset"],
                                     test_set_name_list=csc.config.path["test_set"])
    # test_idx = [74]
    # df = cs_classifier.csd_data_df.iloc[test_idx]
    # X, y = csc.CSClassifier.extract_features_from_df(df)
    # result, label = cs_classifier.predict(input_data=X)
    # print(cs_classifier.classifier.transform(X))
    # print(result)
    # print(y)
    cs_classifier.cross_val_score(42)

    cs_classifier.view_feature()



if __name__ == "__main__":
    main()
