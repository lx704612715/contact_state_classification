import pandas as pd
import os
import numpy as np
from loguru import logger
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn_som.som import SOM
from . import config as cfg
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit


class CSClassifier:
    def __init__(self, experiment_dir, dataset_name):
        self.experiment_dir = experiment_dir
        self.dataset_name = dataset_name
        self.dataset_path = self.experiment_dir + "/csd_result/" + dataset_name + ".pkl"
        self.csd_dataset_plot_dir = self.experiment_dir + "/csd_result/plot/"
        os.makedirs(self.csd_dataset_plot_dir, exist_ok=True)
        self.csc_logger = logger

        # Dataset
        self.csd_data_df = None
        self.csd_data_dict = None
        self.X = []
        self.y = []
        self.X_df = None

        # Classifier
        self.lb = None
        self.classifier = None

        # Dataset information
        self.all_classes = None
        self.num_classes = None

        # Train the classifier
        self.load_data()
        self.setup_classifier()

        self.get_dataset_information()

    def load_data(self):
        # load data to dict, because processing of dataframe takes too much time
        self.csd_data_df = pd.read_pickle(self.dataset_path)
        self.csd_data_dict = self.csd_data_df.to_dict()

    def get_traj_index_by_labels(self, label):
        traj_index_dict = dict()
        for key, value in self.csd_data_dict['label'].items():
            if label in value:
                traj_index_dict[key] = value

        labels = list(set(traj_index_dict.values()))

        return traj_index_dict, labels

    def merge_feature_by_labels(self, traj_index_dict=None, feature="dist", labels=None):
        feature_values = dict()
        for label in labels:
            feature_values[label] = []
            traj_index_list = [key for key, value in traj_index_dict.items() if value == label]
            for traj_index in traj_index_list:
                feature_values[label].append(self.csd_data_dict[feature][traj_index])
        return feature_values

    def setup_classifier(self):
        self.lb = preprocessing.LabelBinarizer()
        self.X, self.y = self.extract_features_from_df(self.csd_data_df.iloc[:74])
        self.X = np.array(self.X)
        self.X_df = pd.DataFrame(data=self.X, index=range(0, self.X.shape[0]),
                                 columns=['act' + str(y) + ' ' + x for x in cfg.params["simple_features"] for y in
                                          range(0, cfg.params["n_act"])])
        y_df = pd.DataFrame(data=self.y, index=range(0, len(self.y)), columns=['label'])
        self.X_df = self.X_df.join(y_df)
        self.lb.fit(self.y)
        # self.y = self.lb.transform(self.y)
        num_labels = np.unique(self.y, axis=0).shape[0]
        if cfg.params["classifier"] == "KNN":
            self.classifier = KNeighborsClassifier(n_neighbors=cfg.params["n_neighbors"])
            self.classifier.fit(self.X, self.y)
        elif cfg.params["classifier"] == "SOM":
            self.classifier = SOM(m=6, n=1, dim=self.X.shape[1])
            self.classifier.fit(self.X, epochs=10, shuffle=False)
        else:
            return

    def cross_val_score(self, random_state=None):
        skf = StratifiedKFold(n_splits=cfg.params["n_splits"], shuffle=True, random_state=random_state)
        print(cross_val_score(self.classifier, self.X, self.y, cv=skf))

    def fit(self):
        if cfg.params["classifier"] == "KNN":
            self.classifier.fit(self.X, self.y)
        elif cfg.params["classifier"] == "SOM":
            self.classifier.fit(self.X, epochs=10, shuffle=False)
        else:
            return

    def get_dataset_information(self):
        self.all_classes = self.lb.classes_
        self.num_classes = len(self.all_classes)
        self.csc_logger.info("All classes from the dataset {} are {}: ", self.dataset_name, self.all_classes)

    def predict(self, input_data):
        result = self.classifier.predict(input_data)
        if cfg.params["classifier"] == "KNN":
            label = self.lb.inverse_transform(result)
            return result, label
        elif cfg.params["classifier"] == "SOM":
            return result, None

    def pca(self, n_components=5):
        pca = PCA(n_components=n_components, svd_solver='auto', whiten='true')
        pca.fit(self.X)
        print(pca.explained_variance_ratio_)
        print(pca.explained_variance_)

    @staticmethod
    def extract_features_from_df(df):
        X = []
        y = []
        for index, row in df.iterrows():
            x = []
            for feature in cfg.params["simple_features"]:
                x = x + row[feature]
            for feature in cfg.params["complex_features"]:
                x = x + np.concatenate(row[feature]).ravel().tolist()
            X.append(x)
            y.append(row["label"])
        X = np.array(X)
        return X, y
