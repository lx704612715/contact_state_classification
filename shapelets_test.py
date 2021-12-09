import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import LearningShapelets, \
    grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size
import contact_state_classification as csc
from tensorflow.keras.optimizers import Adam
from visdom import Visdom
from contact_state_classification import config as cfg
import os


def example():
    # Set seed for determinism
    np.random.seed(0)

    # Load the Trace dataset
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

    # Normalize each of the timeseries in the Trace dataset
    X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
    X_test = TimeSeriesScalerMinMax().fit_transform(X_test)

    # Get statistics of the dataset
    n_ts, ts_sz = X_train.shape[:2]
    n_classes = len(set(y_train))

    # Set the number of shapelets per size as done in the original paper
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                           ts_sz=ts_sz,
                                                           n_classes=n_classes,
                                                           l=0.1,
                                                           r=1)

    # Define the model using parameters provided by the authors (except that we
    # use fewer iterations here)
    shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                optimizer=tf.optimizers.Adam(.01),
                                batch_size=16,
                                weight_regularizer=.01,
                                max_iter=200,
                                random_state=42,
                                verbose=0)
    shp_clf.fit(X_train, y_train)

    # Make predictions and calculate accuracy score
    pred_labels = shp_clf.predict(X_test)
    print("Correct classification rate:", accuracy_score(y_test, pred_labels))

    # Plot the different discovered shapelets
    plt.figure()
    for i, sz in enumerate(shapelet_sizes.keys()):
        plt.subplot(len(shapelet_sizes), 1, i + 1)
        plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
        for shp in shp_clf.shapelets_:
            if ts_size(shp) == sz:
                plt.plot(shp.ravel())
        plt.xlim([0, max(shapelet_sizes.keys()) - 1])

    plt.tight_layout()
    plt.show()

    # The loss history is accessible via the `model_` that is a keras model
    plt.figure()
    plt.plot(np.arange(1, shp_clf.n_iter_ + 1), shp_clf.history_["loss"])
    plt.title("Evolution of cross-entropy loss during training")
    plt.xlabel("Epochs")
    plt.show()


def ex2():
    # Set a seed to ensure determinism
    np.random.seed(42)

    # Load the Trace dataset
    X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")

    # Filter out classes 2 and 4
    mask = np.isin(y_train, [1, 3])
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Normalize the time series
    X_train = TimeSeriesScalerMinMax().fit_transform(X_train)

    # Get statistics of the dataset
    n_ts, ts_sz = X_train.shape[:2]
    n_classes = len(set(y_train))

    # We will extract 1 shapelet and align it with a time series
    shapelet_sizes = {20: 1}

    # Define the model and fit it using the training data
    shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                weight_regularizer=0.001,
                                optimizer=Adam(lr=0.01),
                                max_iter=250,
                                verbose=0,
                                scale=False,
                                random_state=42)
    shp_clf.fit(X_train, y_train)

    # Get the number of extracted shapelets, the (minimal) distances from
    # each of the timeseries to each of the shapelets, and the corresponding
    # locations (index) where the minimal distance was found
    n_shapelets = sum(shapelet_sizes.values())
    distances = shp_clf.transform(X_train)
    predicted_locations = shp_clf.locate(X_train)

    f, ax = plt.subplots(2, 1, sharex=True)

    # Plot the shapelet and align it on the best matched time series. The optimizer
    # will often enlarge the shapelet to create a larger gap between the distances
    # of both classes. We therefore normalize the shapelet again before plotting.
    test_ts_id = np.argmin(np.sum(distances, axis=1))
    shap = shp_clf.shapelets_[0]
    shap = TimeSeriesScalerMinMax().fit_transform(shap.reshape(1, -1, 1)).flatten()
    pos = predicted_locations[test_ts_id, 0]
    ax[0].plot(X_train[test_ts_id].ravel())
    ax[0].plot(np.arange(pos, pos + len(shap)), shap, linewidth=2)
    ax[0].axvline(pos, color='k', linestyle='--', alpha=0.25)
    ax[0].set_title("The aligned extracted shapelet")

    # We calculate the distances from the shapelet to the timeseries ourselves.
    distances = []
    time_series = X_train[test_ts_id].ravel()
    for i in range(len(time_series) - len(shap)):
        distances.append(np.linalg.norm(time_series[i:i + len(shap)] - shap))
    ax[1].plot(distances)
    ax[1].axvline(np.argmin(distances), color='k', linestyle='--', alpha=0.25)
    ax[1].set_title('The distances between the time series and the shapelet')

    plt.tight_layout()
    plt.show()


def main():
    experiment_dir = csc.config.path["experiment_dir"]
    cs_classifier = csc.CSClassifier(experiment_dir=experiment_dir, dataset_name=csc.config.path["dataset_name"])
    # cs_classifier.pca()
    df = cs_classifier.csd_data_df
    X, y = csc.CSClassifier.extract_features_from_df_for_shapelet(df)

    # Normalize each of the timeseries in the Trace dataset
    # X_train = TimeSeriesScalerMinMax().fit_transform(X)

    # Get statistics of the dataset
    n_ts, ts_sz = X.shape[:2]
    n_classes = len(set(y))

    # Set the number of shapelets per size as done in the original paper
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                           ts_sz=ts_sz,
                                                           n_classes=n_classes,
                                                           l=1,
                                                           r=1)
    print(shapelet_sizes)
    # Define the model using parameters provided by the authors (except that we
    # use fewer iterations here)
    shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                optimizer=tf.optimizers.Adam(.01),
                                batch_size=16,
                                weight_regularizer=.01,
                                max_iter=400,
                                random_state=42,
                                verbose=0)
    shp_clf.fit(X, y)
    distances = shp_clf.transform(X)
    # Make predictions and calculate accuracy score
    pred_labels = shp_clf.predict(X)
    print("Correct classification rate:", accuracy_score(y, pred_labels))

    # Plot the different discovered shapelets
    plt.figure()
    for i, sz in enumerate(shapelet_sizes.keys()):
        plt.subplot(len(shapelet_sizes), 1, i + 1)
        plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
        for shp in shp_clf.shapelets_:
            if ts_size(shp) == sz:
                plt.plot(shp.ravel())
        plt.xlim([0, max(shapelet_sizes.keys()) - 1])

    plt.tight_layout()
    plt.show()

    # The loss history is accessible via the `model_` that is a keras model
    plt.figure()
    plt.plot(np.arange(1, shp_clf.n_iter_ + 1), shp_clf.history_["loss"])
    plt.title("Evolution of cross-entropy loss during training")
    plt.xlabel("Epochs")
    plt.show()

    viz = Visdom()
    assert viz.check_connection()
    try:
        viz.scatter(
            X=distances,
            Y=[cfg.params["cs_index_map"][x] for x in pred_labels],
            opts=dict(
                legend=list(cfg.params["cs_index_map"].keys()),
                markersize=5,
                xlabel="Distance to 1st Shapelet",
                ylabel="Distance to 2nd Shapelet",
                zlabel="Distance to 3rd Shapelet",
            )
        )

    except BaseException as err:
        print('Skipped matplotlib example')
        print('Error message: ', err)

if __name__ == "__main__":
    example()
