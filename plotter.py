import csv
from visdom import Visdom
import numpy as np
from contact_state_classification import config as cfg


def make_plot(index_list):
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
            features = row[7]
            kernel_size = row[10]
            legends.append(row[4])

    viz = Visdom()
    assert viz.check_connection()
    try:
        viz.boxplot(
            X=X,
            opts=dict(
                legend=legends,
                title="KNN score using %s features with kernel size of %s" % (features, kernel_size),
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
    with open('./scores.csv') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            scores.append(float(row[9]))
            rownames.append(str(i) + ' KNN ' + str(len(list(row[7].strip('][').split(', ')))) + ' features')
    with open('./scores_SHP.csv') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            scores.append(float(row[9]))
            rownames.append(str(i) + ' SHP ' + str(len(list(row[7].strip('][').split(', ')))) + ' features ' + row[10] + ' kernel size')

    viz = Visdom()
    assert viz.check_connection()
    try:
        viz.bar(
            X=np.array(scores),
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
    # make_plot([0, 1, 2])
    # make_plot([3, 4, 5])
    # make_plot([6, 7, 8])
    # Kernel size decide weather we want local exploitation or global exploration
    # make_plot([12, 13, 14])
    plot_result_from_diff_grasp_pose()


if __name__ == "__main__":
    main()
