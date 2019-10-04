import urllib
import os
from os import path
from random import sample

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D   # needed

import pandas as pd
import seaborn as sns
import numpy as np
import operator

# files import
from misc_libary import *

# visualize data
def visualize_Data(df: object) -> None:
    """
    :type df: Dataframe from pandas
    """
    sns.set_style("darkgrid")

    # set number of subplots and size
    axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    axes = axes[1].flatten()  # axes[1] because axes is a tulpl and figure is in it

    # draw kdeplot
    sns.kdeplot(df["MEDV"], shade=True, cut=0, ax=axes[0], color="blue")
    plt.xlabel("MEDV")
    plt.ylabel("probapility")
    axes[0].title.set_text("Distribution of MEDV Values")


    # draw scatterplot
    sns.scatterplot(x=df["LSTAT"], y=df["MEDV"], color="green", ax=axes[1])
    plt.xlabel("LSTAT")
    plt.ylabel("MEDV")
    axes[1].title.set_text("Medv in Relation to Lstat")


    # convert our dataframe into a rounded correlation matrix
    correlation_matrix = df.corr().round(2)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(correlation_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    # draw the heatmap
    sns.heatmap(data=correlation_matrix, mask=mask, annot=True, cmap="coolwarm", square=True, ax=axes[2])
    axes[2].title.set_text("Correlation-Matrix of the Data")


    # draw the kdeplot
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    sns.kdeplot(df["RM"], df["MEDV"], cmap=cmap, n_levels=8, shade=True, ax=axes[3])
    axes[3].title.set_text("Medv in Relation to RM")


def v_model_poly(x_axis, y_axis, weights, data_train, target_train, args):
    # create our figure. With size of the figure and specifying the art of diagrams we use "3d"
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca(projection='3d')

    # data gets created for visualizing our model
    if args.model == "polynomial_regression":
        f1 = np.arange(-0.7, 1.2, 0.1)
        f2 = np.arange(-0.7, 1.2, 0.1)
        f3 = np.arange(-0.7, 1.2, 0.1)

    elif args.model == "normal_equation":
        f1 = np.arange(4, 8, 0.17)  # RM
        f2 = np.arange(11, 35, 1)  # LSTAT
        f3 = np.arange(14, 22, 0.34)  # PTRATIO

        print(len(f1))
        print(len(f2))
        print(len(f3))
        print(" ")

    f1, f2 = np.meshgrid(f1, f2)
    # z corosponds to medv
    # hypothesis_pol(weights, f1, f2, f3, bias):
    # Z = weights[0] * f1 + weights[1] * f1 ** 2 + weights[2] * f2 + weights[3] * f2 ** 2 + \
    #     weights[4] * f3 + \
    #     weights[5] * f3 ** 2 + weights[6] * 1

    if args.model == "polynomial_regression":
        Z = hypothesis_pol(weights, f1, f2, f3, 1)
    elif args.model == "normal_equation":
        Z = hypothesis_normal(weights, f1, f2, f3, 1)

    # ploting our model
    ax.plot_surface(f1, f2, Z, alpha=0.3, edgecolors='grey')

    # ploting our data points from our dataframe
    if args.model == "polynomial_regression":
        X = data_train[x_axis]
        Y = data_train[y_axis]
        Z = target_train

    elif args.model == "normal_equation":
        X = data_train[x_axis]
        Y = data_train[y_axis]
        Z = target_train

    ax.scatter3D(X, Y, Z, c=Z, s=40, alpha=0.9)  # cmap=cm.coolwarm

    # change the inital view point
    ax.view_init(azim=30)

    # title
    ax.set_title("Descision border and datapoints")

    # set label descrtiption
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel("MEDV")

    # hide the ticks of the label
    #ax.axes.xaxis.set_ticklabels([])
    #ax.axes.yaxis.set_ticklabels([])
    #ax.axes.zaxis.set_ticklabels([])

    # hide the grid
    ax.grid(False)


# visualize our model. the function visualize() is not in the class model so that we can use multiprocessing.
def visualize(args, df_data, parameter_list: list) -> None:

    # unzip the argument list gotten from model.getter_viszulation()
    if args.model == "linear_regression" or args.model == "polynomial_regression":
        weights = parameter_list[0]
        train_loss_history = parameter_list[1]
        test_loss_history = parameter_list[2]
        evaluation_time = parameter_list[3]
        data_train = parameter_list[4]
        target_train = parameter_list[5]
        x_train_loose = parameter_list[6]

        # test data
        data_test = df_data[1]["RM"].tolist()
        target_test = df_data[1]["MEDV"].tolist()

    elif args.model == "normal_equation":
        weights = parameter_list[0]
        evaluation_time = parameter_list[1]

    if args.model == "normal_equation":
        # train data
        data_train_normal = df_data.iloc[:,  df_data.columns != "MEDV"]   # get all eleements from the df except "medv"
        target_train_normal = df_data["MEDV"].tolist()

    # prints Mean loss of last epoch
    if not args.infile and not args.model == "normal_equation":
        print(" ")
        print("Mean-train loss of last epoch: ", str(round(train_loss_history[-1], 6)))
        print("Mean-test loss of last epoch:  ", str(round(test_loss_history[-1], 6)))
        print("Time needed for training:      ", str(round(evaluation_time, 4)) + "s.")

    # communication is key
    if args.fd == "full" and not args.infile:
        print(" ")
        print("-----------------------------------------------------------")
        # print for every value in the list weights
        for i in range(len(weights)):
            print("Value of W" + str(i) + " after training:   ", weights[i])

        print("-----------------------------------------------------------")
        print(" ")
    elif args.infile:
        print(" ")
        # print for every value in the list weights
        for i in range(len(weights)):
            print("Value of W" + str(i) + " after training:   ", weights[i])

        print(" ")

    # visualize data if argument says so
    if args.v_data:
        visualize_Data(df_data)

    # get points for line
    X = []
    Y = []
    for i in range(5, 20):
        X.append(i * 0.1)
        Y.append(weights[0] * (i * 0.1) + weights[1])


    # plot our descion border and datapoints
    if args.v_model:
        if args.model == "linear_regression":
            sns.set_style("darkgrid")
            plt.figure(figsize=(9, 6))
            plt.title("Decision Border and Data-points")
            plt.xlabel("Average number of rooms per dwelling(Wohnung)")
            plt.ylabel("Median value of owner-occupied homes in 1000$")
            sns.lineplot(x=X, y=Y)
            sns.scatterplot(x=data_train, y=target_train, color="orange", label="train data")
            sns.scatterplot(x=data_test, y=target_test, color="green", label="test data")
            plt.legend()

        elif args.model == "polynomial_regression":
            v_model_poly("RM", "LSTAT", weights, data_train, target_train, args)
            v_model_poly("RM", "PTRATIO", weights, data_train, target_train, args)

        elif args.model == "normal_equation":
            v_model_poly("RM", "LSTAT", weights, data_train_normal, target_train_normal, args)
            v_model_poly("RM", "PTRATIO", weights, data_train_normal, target_train_normal, args)

    # convert our loss arrays into a dataframe from pandas
    if not args.model == "normal_equation":
        data = {"x": x_train_loose, "train": train_loss_history, "test": test_loss_history}
        data = pd.DataFrame(data, columns=["x", "train", "test"])

    # plot loss over time
    if args.v_loss and not args.infile:
        if not args.model == "normal_equation":
            sns.set_style("darkgrid")
            plt.figure(figsize=(12, 6))
            sns.lineplot(x="x", y="train", data=data, label="train", color="orange")
            sns.lineplot(x="x", y="test", data=data, label="test", color="Violet")
            plt.xlabel("Time In Epochs")
            plt.ylabel("Loss")
            plt.title("Loss over Time")

        else:
            print(" ")
            print("This model doesn't support this feature.")

    # plt.show() when we have a diagram
    if args.v_loss or args.v_model or args.v_data:
        plt.show()
