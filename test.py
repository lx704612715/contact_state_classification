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


def main():
    experiment_dir = csc.config.path["experiment_dir"]
    cs_classifier = csc.CSClassifier(experiment_dir=experiment_dir, dataset_name=csc.config.path["dataset_name"])
    # test_idx = [74]
    # df = cs_classifier.csd_data_df.iloc[test_idx]
    # X, y = csc.CSClassifier.extract_features_from_df(df)
    # result, label = cs_classifier.predict(input_data=X)
    # print(cs_classifier.classifier.transform(X))
    # print(result)
    # print(y)
    cs_classifier.cross_val_score(42)
    # Plot the distances
    viz = Visdom()
    assert viz.check_connection()
    try:
        viz.scatter(
            X=cs_classifier.X,
            Y=[cfg.params["cs_index_map"][x] for x in cs_classifier.y],
            opts=dict(
                legend=list(cfg.params["cs_index_map"].keys()),
                markersize=10,
            )
        )
    except BaseException as err:
        print('Skipped matplotlib example')
        print('Error message: ', err)


if __name__ == "__main__":
    main()
