import csv
from visdom import Visdom
import numpy as np
from contact_state_classification import config as cfg


def make_knn_plot(index_list, title="KNN score using different features"):
    X = np.zeros((cfg.N_SPLITS, 0))
    legends = []
    features = None
    kernel_size = None
    with open('./scores.csv') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i not in index_list:
                continue
            x_slice = list(map(float, row[8].strip('][').split(', ')))
            X = np.hstack((X, np.array(x_slice).reshape((len(x_slice), 1))))
            features = list(row[7].strip('][').split(', '))
            legends.append("Using %d feature(s)" % len(features))

    viz = Visdom()
    assert viz.check_connection()

    try:
        viz.boxplot(
            X=X,
            env="KNN",
            opts=dict(
                legend=legends,
                title=title,
                ylabel="Accuracy",
                ytickmin=0,
                ytickmax=1,
            )
        )
    except BaseException as err:
        print('Skipped matplotlib example')
        print('Error message: ', err)

def make_shp_plot(index_list):
    X = np.zeros((cfg.N_SPLITS, 0))
    legends = []
    features = None
    kernel_size = None
    with open('./scores_SHP.csv') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i not in index_list:
                continue
            x_slice = list(map(float, row[8].strip('][').split(', ')))
            X = np.hstack((X, np.array(x_slice).reshape((len(x_slice), 1))))
            features = list(row[7].strip('][').split(', '))
            kernel_size = row[10]
            legends.append(row[4])

    viz = Visdom()
    assert viz.check_connection()
    try:
        viz.boxplot(
            X=X,
            env="SHP",
            opts=dict(
                legend=legends,
                title="SHP score using %d features with kernel size of %s" % (len(features), kernel_size),
                ylabel="Accuracy",
                xlabel="Interpolation's method",
                ytickmin=0,
                ytickmax=1,
            )
        )
    except BaseException as err:
        print('Skipped matplotlib example')
        print('Error message: ', err)

def plot_result_from_diff_grasp_pose():
    scores = []
    rownames = []
    j = 1
    with open('./scores.csv') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            scores.append(float(row[9]))
            rownames.append(str(j) + ' KNN ' + str(len(list(row[7].strip('][').split(', ')))) + ' features')
            j += 1
    with open('./scores_SHP.csv') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            scores.append(float(row[9]))
            rownames.append(str(j) + ' SHP ' + str(len(list(row[7].strip('][').split(', ')))) + ' features ' + row[10] + ' kernel size')
            j += 1
    viz = Visdom()
    assert viz.check_connection()
    try:
        viz.bar(
            X=np.array(scores),
            env="main",
            opts=dict(
                stacked=True,
                rownames=rownames,
                title="Test score using data from different grasp pose",
                ytickmin=0,
                ytickmax=1,
            )
        )
    except BaseException as err:
        print('Skipped matplotlib example')
        print('Error message: ', err)


def main():
    # Bad feature selection can make result worse
    make_knn_plot([0, 1, 2])
    make_knn_plot([3, 4, 5], "KNN score using different features with PCA")
    make_shp_plot([0, 1, 2])
    make_shp_plot([3, 4, 5])
    make_shp_plot([6, 7, 8])
    # Kernel size decide weather we want local exploitation or global exploration
    make_shp_plot([12, 13, 14])
    plot_result_from_diff_grasp_pose()


if __name__ == "__main__":
    main()
