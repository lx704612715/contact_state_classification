import csv
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from scipy.interpolate import interp1d
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn_som.som import SOM
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import LearningShapelets, \
    grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size
from visdom import Visdom

import contact_state_classification as csc
from . import config as cfg


class CSClassifier:
    def __init__(self, experiment_dir, dataset_name_list, test_set_name_list):
        self.experiment_dir = experiment_dir
        self.dataset_name_list = dataset_name_list
        self.dataset_path_list = [self.experiment_dir + "/csd_result/" + x + ".pkl" for x in dataset_name_list]
        self.test_set_list = [self.experiment_dir + "/csd_result/" + x + ".pkl" for x in test_set_name_list]
        self.csd_dataset_plot_dir = self.experiment_dir + "/csd_result/plot/"
        os.makedirs(self.csd_dataset_plot_dir, exist_ok=True)
        self.csc_logger = logger

        # Dataset
        self.csd_data_df = None
        self.csd_data_dict = None
        self.csd_test_data_df = None
        self.csd_test_data_dict = None
        # Filtered data for training
        self.X = []
        self.y = []
        # Filtered data for visualization
        self.X_df = None

        self.pca = None

        # Classifier
        self.lb = None
        self.classifier = None

        # Dataset information
        self.all_classes = None
        self.num_classes = None

        # Train the classifier
        self.load_data(cfg.CIRCULAR_SPLICING)
        self.setup_classifier(cfg.USE_PCA)

        self.get_dataset_information()

    def load_data(self, circular_splicing):
        # load data to dict, because processing of dataframe takes too much time
        for path in self.dataset_path_list:
            if self.csd_data_df is None:
                self.csd_data_df = pd.read_pickle(path)
            else:
                self.csd_data_df.append(pd.read_pickle(path))
        self.csd_data_dict = self.csd_data_df.to_dict()

        for path in self.test_set_list:
            if self.csd_test_data_df is None:
                self.csd_test_data_df = pd.read_pickle(path)
            else:
                self.csd_test_data_df.append(pd.read_pickle(path))
        if len(self.test_set_list) != 0:
            self.csd_test_data_dict = self.csd_test_data_df.to_dict()

        self.lb = preprocessing.LabelBinarizer()
        if cfg.CLASSIFIER == "SHP":
            self.X, self.y = csc.CSClassifier.extract_features_from_df_for_shapelet(self.csd_data_df, circular_splicing)
        else:
            self.X, self.y = self.extract_features_from_df(self.csd_data_df)
        self.lb.fit(self.y)

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

    def setup_classifier(self, use_pca=False):
        # self.y = self.lb.transform(self.y)
        num_labels = np.unique(self.y, axis=0).shape[0]
        if use_pca:
            self.apply_pca()
        if cfg.CLASSIFIER == "KNN":
            self.classifier = KNeighborsClassifier(n_neighbors=cfg.N_NEIGHBORS)
            self.classifier.fit(self.X[:101], self.y[:101])
            if cfg.BASIC_VISUALIZATION:
                self.knn_visualization()
        elif cfg.CLASSIFIER == "SOM":
            self.classifier = SOM(m=6, n=1, dim=self.X.shape[1])
            self.classifier.fit(self.X, epochs=10, shuffle=False)
        elif cfg.CLASSIFIER == "SHP":
            n_ts, ts_sz = self.X.shape[:2]
            n_classes = len(set(self.y))

            # Set the number of shapelets per size as done in the original paper
            shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                                   ts_sz=ts_sz,
                                                                   n_classes=n_classes,
                                                                   l=cfg.SHAPELET_SIZE,
                                                                   r=1)
            print(shapelet_sizes)
            # Define the model using parameters provided by the authors (except that we
            # use fewer iterations here)
            self.classifier = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                                optimizer=tf.optimizers.Adam(.01),
                                                batch_size=16,
                                                weight_regularizer=.01,
                                                max_iter=800,
                                                random_state=42,
                                                verbose=0)
            self.X = TimeSeriesScalerMinMax().fit_transform(self.X)
            self.classifier.fit(self.X,
                                self.y)
            # Make predictions and calculate accuracy score
            pred_labels = self.classifier.predict(TimeSeriesScalerMinMax().fit_transform(self.X))
            print("Correct classification rate:", accuracy_score(self.y, pred_labels))

            if cfg.BASIC_VISUALIZATION:
                self.view_feature()
                self.shapelet_visualization(shapelet_sizes)

        else:
            return

    def knn_visualization(self):
        # Plot the distances
        viz = Visdom()
        assert viz.check_connection()
        try:
            viz.scatter(
                X=self.X,
                Y=[cfg.CS_INDEX_MAP[x] for x in self.y],
                opts=dict(
                    legend=list(cfg.CS_INDEX_MAP.keys()),
                    markersize=10,
                    title="After PCA with %d PC" % cfg.N_COMPONENTS,
                    xlabel="PC1",
                    ylabel="PC2",
                    zlabel="PC3",
                )
            )
        except BaseException as err:
            print('Skipped matplotlib example')
            print('Error message: ', err)

    def shapelet_visualization(self, shapelet_sizes):
        distances = self.classifier.transform(self.X)
        viz = Visdom()
        assert viz.check_connection()
        try:
            # for i, sz in enumerate(shapelet_sizes.keys()):
            #     viz.scatter(
            #         X=distances,
            #         Y=[cfg.params["cs_index_map"][x] for x in pred_labels],
            #         opts=dict(
            #             legend=list(cfg.params["cs_index_map"].keys()),
            #             markersize=5,
            #             title="%d shapelets of size %d" % (shapelet_sizes[sz], sz),
            #             xlabel="Distance to 1st Shapelet",
            #             ylabel="Distance to 2nd Shapelet",
            #             zlabel="Distance to 3rd Shapelet",
            #         )
            #     )

            for i, sz in enumerate(shapelet_sizes.keys()):
                shapelets = np.zeros((sz, 0))
                for shp in self.classifier.shapelets_:
                    if ts_size(shp) == sz:
                        shapelets = np.hstack((shapelets, shp.ravel().reshape((sz, 1))))

                viz.line(
                    X=np.linspace(0, sz, num=sz, endpoint=False),
                    Y=shapelets,
                    opts=dict(
                        markersize=10,
                        title="%d shapeplets of size %d" % (shapelet_sizes[sz], sz),
                        xlabel="ACT",
                        ylabel="Dist"
                    )
                )

                viz.line(
                    X=np.arange(1, self.classifier.n_iter_ + 1),
                    Y=self.classifier.history_["loss"],
                    opts=dict(
                        markersize=10,
                        title="%d shapeplets of size %d" % (shapelet_sizes[sz], sz),
                        xlabel="Epochs",
                        ylabel="Loss"
                    )
                )

        except BaseException as err:
            print('Skipped matplotlib example')
            print('Error message: ', err)

    def view_feature(self):
        viz = Visdom()
        assert viz.check_connection()
        new_dim = csc.config.N_ACT * csc.config.UPSAMPLING_RATE
        if csc.config.CIRCULAR_SPLICING:
            new_dim = new_dim * 2
        new_x = np.linspace(1, csc.config.N_ACT,
                            num=new_dim,
                            endpoint=True)

        try:
            for cs in csc.config.CS_INDEX_MAP.keys():
                index_list = np.where(self.y == cs)[0].tolist()
                viz.line(
                    X=new_x,
                    Y=np.take(self.X, index_list, 0)[:, :, 0].T,
                    opts=dict(
                        markersize=10,
                        xtick=True,
                        xtickstep=1,
                        title="Distance on " + cs,
                        xlabel="ACT",
                        ylabel="Distance"
                    )
                )
        except BaseException as err:
            print('Skipped matplotlib example')
            print('Error message: ', err)

    def extract_df(self):
        X, y = self.extract_features_from_df(self.csd_data_df)
        columns_simple_features = ['act' + str(y) + ' ' + x for x in cfg.SIMPLE_FEATURES for y in
                                   range(0, cfg.N_ACT)]
        columns_complex_features = ['act_' + str(y) + ' joint_' + str(z) + ' ' + x for x in
                                    cfg.COMPLEX_FEATURES for y in
                                    range(0, cfg.N_ACT) for z in range(self.csd_data_dict[x][0][0].shape[0])]
        X_df = pd.DataFrame(data=X, index=range(0, X.shape[0]),
                            columns=columns_simple_features + columns_complex_features)
        y_df = pd.DataFrame(data=self.y, index=range(0, len(self.y)), columns=['label'])
        return X_df.join(y_df)

    def cross_val_score(self, random_state=None):
        skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=random_state)
        if cfg.CLASSIFIER == "KNN":
            return cross_val_score(self.classifier, self.X, self.y, cv=skf).tolist()
        elif cfg.CLASSIFIER == "SOM":
            return
        elif cfg.CLASSIFIER == "SHP":
            score = []
            for train_index, test_index in skf.split(self.X, self.y):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
                X_test = TimeSeriesScalerMinMax().fit_transform(X_test)
                self.classifier.fit(X_train, y_train)
                pred_labels = self.classifier.predict(X_test)
                # print(self.classifier.predict_proba(X_test))
                s = accuracy_score(y_test, pred_labels)
                print(s)
                score.append(s)
            return score
        else:
            return

    def score_with_diff_grasp_pose(self):
        if cfg.CLASSIFIER == "KNN":
            X_test, y_test = self.extract_features_from_df(self.csd_test_data_df)
            if cfg.USE_PCA:
                X_test = self.pca.transform(X_test)
            pred_labels = self.classifier.predict(X_test)
            return accuracy_score(y_test, pred_labels)
        elif cfg.CLASSIFIER == "SOM":
            return
        elif cfg.CLASSIFIER == "SHP":
            X_test, y_test = self.extract_features_from_df_for_shapelet(self.csd_test_data_df,
                                                                        cfg.CIRCULAR_SPLICING)
            X_test = TimeSeriesScalerMinMax().fit_transform(X_test[101:])
            pred_labels = self.classifier.predict(X_test)
            # print(self.classifier.predict_proba(X_test))
            return accuracy_score(y_test[101:], pred_labels)

    def log_to_csv(self, random_state=None, file_path=None):
        columns_KNN = ["Classifier",
                       "Use PCA",
                       "Upsampling Rate",
                       "Circular Splicing",
                       "Interpolation Method",
                       "Training Dataset",
                       "Test Dataset",
                       "Features",
                       "Scores from same Dataset",
                       "Scores from other Dataset"]

        columns_SHP = ["Classifier",
                       "Use PCA",
                       "Upsampling Rate",
                       "Circular Splicing",
                       "Interpolation Method",
                       "Training Dataset",
                       "Test Dataset",
                       "Features",
                       "Scores from same Dataset",
                       "Scores from other Dataset",
                       "Shapelet Size"]

        if os.path.isfile(file_path):
            csvfile = open(file_path, 'a')
        else:
            csvfile = open(file_path, 'w')
        csvwriter = csv.writer(csvfile)
        if cfg.CLASSIFIER == "SHP":
            csvwriter.writerow([cfg.CLASSIFIER,
                                cfg.USE_PCA,
                                cfg.UPSAMPLING_RATE,
                                cfg.CIRCULAR_SPLICING,
                                cfg.INTERPOLATION_METHOD,
                                cfg.path["dataset"],
                                cfg.path["test_set"],
                                cfg.SIMPLE_FEATURES + cfg.COMPLEX_FEATURES,
                                str(self.cross_val_score(random_state)),
                                str(self.score_with_diff_grasp_pose()),
                                cfg.SHAPELET_SIZE])
        elif cfg.CLASSIFIER == "KNN":
            csvwriter.writerow([cfg.CLASSIFIER,
                                cfg.USE_PCA,
                                cfg.UPSAMPLING_RATE,
                                cfg.CIRCULAR_SPLICING,
                                cfg.INTERPOLATION_METHOD,
                                cfg.path["dataset"],
                                cfg.path["test_set"],
                                cfg.SIMPLE_FEATURES + cfg.COMPLEX_FEATURES,
                                str(self.cross_val_score(random_state)),
                                str(self.score_with_diff_grasp_pose())])
        csvfile.close()

    def fit(self):
        if cfg.CLASSIFIER == "KNN":
            self.classifier.fit(self.X, self.y)
        elif cfg.CLASSIFIER == "SOM":
            self.classifier.fit(self.X, epochs=10, shuffle=False)
        else:
            return

    def get_dataset_information(self):
        self.all_classes = self.lb.classes_
        self.num_classes = len(self.all_classes)
        self.csc_logger.info("All classes from the dataset {} are {}: ", self.dataset_name_list, self.all_classes)

    def predict(self, input_data):
        result = self.classifier.predict(input_data)
        if cfg.CLASSIFIER == "KNN":
            label = self.lb.inverse_transform(result)
            return result, label
        elif cfg.CLASSIFIER == "SOM":
            return result, None

    def apply_pca(self):
        if len(self.X.shape) > 2:
            return
        self.pca = PCA(n_components=cfg.N_COMPONENTS, svd_solver='auto', whiten='true')
        self.pca.fit(self.X[:101])
        self.X = self.pca.transform(self.X[:101])
        print("variance_ratio:")
        print(self.pca.explained_variance_ratio_)
        print("variance:")
        print(self.pca.explained_variance_)

    @staticmethod
    def extract_features_from_df(df):
        X = []
        y = []
        for index, row in df.iterrows():
            x = []
            for feature in cfg.SIMPLE_FEATURES:
                x = x + row[feature]
            for feature in cfg.COMPLEX_FEATURES:
                x = x + np.concatenate(row[feature]).ravel().tolist()
            X.append(x)
            y.append(row["label"])
        X = np.array(X)
        y = np.array(y)
        return X, y

    @staticmethod
    def extract_features_from_df_for_shapelet(df, circular_splicing):
        X = []
        y = []
        new_dim = csc.config.N_ACT * csc.config.UPSAMPLING_RATE
        n_act = csc.config.N_ACT
        if circular_splicing:
            new_dim = new_dim * 2
            n_act = n_act * 2
        for index, row in df.iterrows():
            x = np.zeros((new_dim, 0))
            for feature in cfg.SIMPLE_FEATURES:
                original_y = np.array(row[feature]).reshape((cfg.N_ACT))
                original_x = np.linspace(1, n_act, num=n_act, endpoint=True)
                if circular_splicing:
                    original_y = np.hstack((original_y, original_y))
                f = interp1d(original_x, original_y, kind=csc.config.INTERPOLATION_METHOD)
                new_x = np.linspace(1, n_act, num=new_dim, endpoint=True)
                x_slice = f(new_x)
                x = np.hstack((x, x_slice.reshape((x_slice.size, 1))))
            for feature in cfg.COMPLEX_FEATURES:
                original_y = np.array(row[feature]).T
                original_x = np.linspace(1, n_act, num=n_act, endpoint=True)
                new_x = np.linspace(1, n_act, num=new_dim, endpoint=True)
                if circular_splicing:
                    original_y = np.hstack((original_y, original_y))
                for r in original_y:
                    f = interp1d(original_x, r, kind=csc.config.INTERPOLATION_METHOD)
                    x_slice = f(new_x)
                    x = np.hstack((x, x_slice.reshape((x_slice.size, 1))))
            X.append(x)
            y.append(row["label"])
        X = np.array(X)
        y = np.array(y)
        return X, y
