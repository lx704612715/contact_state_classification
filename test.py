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
from contact_state_classification import config as cfg
import seaborn as sns
import random
import matplotlib.pyplot as plt


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
    plt.subplot(1, 1, 1)
    for color, label in zip('rgbck', ('CS1', 'CS2', 'CS3', 'CS5', 'CS6')):
        plt.scatter(cs_classifier.X[cs_classifier.y == label, 0], cs_classifier.X[cs_classifier.y == label, 1],
                    c=color, label='{}'.format(label))
    plt.title('Point Cloud after PCA Transformation with 2 PC',
              fontsize=14)
    plt.xlabel("1st PC")
    plt.ylabel("2nd PC")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
