import urllib
import os
from os import path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D   # needed

import pandas as pd
import seaborn as sns
import numpy as np
import operator
# # global variables
temp_change = 0

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def loss(pred_target: float, real_traget: float) -> float:
    return round(float((pred_target - real_traget) ** 2), 4)


def get_Data() -> object:
    try:
        if path.isfile("boston_housing.csv"):
            df = pd.read_csv("boston_housing.csv")
            return df
    except:
        try:
            if path.isfile("housing.csv"):
                df = pd.read_csv("housing.csv")
                return df
        except FileNotFoundError:
            print("oops, file doesn't exist")


# used to remove trailing whitespace from file
def chomped_lines(it):
    return map(operator.methodcaller('rstrip', '\r\n'), it)

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


def is_non_zero_file() -> object:
    return (os.path.isfile("housing.csv") and os.path.getsize("housing.csv") > 0) or (
            os.path.isfile("boston_housing.csv") and os.path.getsize("boston_housing.csv") > 0)


def download_dataset() -> None:
    try:
        url = "https://raw.githubusercontent.com/udacity/machine-learning/master/projects/boston_housing/housing.csv"
        if url.lower().startswith('http'):
            file = open("boston_housing.csv", "w+")  # the plus asgins when not exist we create it
            data = urllib.request.urlopen(url).read().decode(
            'utf-8')
            file.write(data)
            file.close()
        else:
            raise ValueError from None
    except:
        try:
            url = "https://raw.githubusercontent.com/LuposX/BostonHousingPrediction/master/dataset/boston_housing.csv"
            if url.lower().startswith('http'):
                file = open("boston_housing.csv", "w+")  # the plus asgins when not exist we create it
                data = urllib.request.urlopen(url).read().decode(
                'utf-8')
                file.write(data)
                file.close()
            else:
                raise ValueError from None
        except Exception as e:
            print("Error: ", str(e))


def preproc_data(df: object, args) -> list:
    # money in 10,000
    df[["MEDV"]] = df[["MEDV"]] / 10000

    if args.model == "linear_regression":
        df_new = df

        # normalization variables for linear regression
        df_new_range = df_new.max() - df_new.min()
        df_new_mean = df_new.std(ddof=1)

        # normalization
        df_new = (df_new - df_new_mean) / df_new_range

        # shuffling data
        df_new = df_new.sample(frac=1).reset_index(drop=True)

        # split in training and test data
        df_new_train = df_new[:380]
        df_new_test = df_new[381:]

        return df_new_train, df_new_test, df_new_range, df_new_mean

    elif args.model == "polynomial_regression" or args.model == "normal_equation":
        df_new = df

        # normalization variables for polynomial regression
        df_new_range = df_new.max() - df_new.min()
        df_new_mean = df_new.mean()

        # normalization
        df_new = (df_new - df_new_mean) / df_new_range

        # shuffling data
        df_new = df_new.sample(frac=1).reset_index(drop=True)

        # split in training and test data
        df_new_train = df_new[:380]
        df_new_test = df_new[381:]

        return df_new_train, df_new_test, df_new_range, df_new_mean

    else:
        print("something went wrong in data preprocessing.")


def hypothesis_pol(weights, f1, f2, f3, bias):
    pred = weights[0] * f1 + weights[1] * f1 ** 2 + weights[2] * f1 ** 3 + weights[3] * f1 ** 4 + \
           weights[4] * f2 + weights[5] * f2 ** 2 + weights[6] * f2 ** 3 + weights[7] * f2 ** 4 + \
           weights[8] * f3 + weights[9] * f3 ** 2 + weights[10] * f3 ** 3 + weights[11] * f3 ** 4 + \
           weights[12] * bias

    return pred


# our hypothesis/ what our model predicts
def hypothesis_normal(weights, f1, f2, f3, bias):
    pred = weights[0] * f1 + weights[1] * f1 ** 2 + weights[2] * f1 ** 3 + \
           weights[3] * f2 + weights[4] * f2 ** 2 + weights[5] * f2 ** 3 + \
           weights[6] * f3 + weights[7] * f3 ** 2 + weights[8] * f3 ** 3 + \
           weights[9] * bias

    return pred

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
    elif args.model == "normal_equation":
        weights = parameter_list[0]
        evaluation_time = parameter_list[1]

    if args.model == "normal_equation":
        # train data
        data_train = df_data[0].iloc[:,  df_data[0].columns != "MEDV"]   # get all eleements from the df except "medv"
        target_train = df_data[0]["MEDV"].tolist()

    # test data
    data_test = df_data[1]["RM"].tolist()
    target_test = df_data[1]["MEDV"].tolist()

    # prints Mean loss of last epoch
    if not args.infile and not args.model == "normal_equation":
        print(" ")
        print("Mean-train loss of last epoch: ", str(round(train_loss_history[-1], 6)))
        print("Mean-test loss of last epoch:  ", str(round(test_loss_history[-1], 6)))
        print("Time needed for training:      ", str(round(evaluation_time, 4)) + "s.")

    # elif args.model == "normal_equation":
    #     loss =
    #     print(" ")
    #     print("Mean-train loss: ", str(round(, 6)))
    #     print("Mean-test loss:  ", str(round(, 6)))
    #     print("Time needed for training:      ", str(round(evaluation_time, 4)) + "s.")

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
            v_model_poly("RM", "LSTAT", weights, data_train, target_train, args)
            v_model_poly("RM", "PTRATIO", weights, data_train, target_train, args)

    # convert our loss arrays into a dataframe from pandas
    if not args.model == "normal_equation":
        data = {"x": x_train_loose, "train": train_loss_history, "test": test_loss_history}
        data = pd.DataFrame(data, columns=["x", "train", "test"])

    # plot loss over time
    if args.v_loss and not args.infile:
        sns.set_style("darkgrid")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="x", y="train", data=data, label="train", color="orange")
        sns.lineplot(x="x", y="test", data=data, label="test", color="Violet")
        plt.xlabel("Time In Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Time")

    # plt.show() when we have a diagram
    if args.v_loss or args.v_model or args.v_data:
        plt.show()


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
        f1 = np.arange(-0.7, 1.2, 0.1)
        f2 = np.arange(-0.7, 1.2, 0.1)
        f3 = np.arange(-0.7, 1.2, 0.1) 

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
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    # hide the grid
    ax.grid(False)


# convert our arguments from strings into booleans
def parse_bool_args(args):
    if args.predict_on == "False" or args.predict_on == "false" or args.predict_on == "false ":
        args.predict_on = False
    if args.predict_on == "True" or args.predict_on == "true" or args.predict_on == "true ":
        args.predict_on = True

    if args.h_features == "False" or args.h_features == "false" or args.h_features == "false ":
        args.h_features = False
    if args.h_features == "True" or args.h_features == "true" or args.h_features == "true ":
        args.h_features = True

    if args.save == "False" or args.save == "false" or args.save == "false ":
        args.save = False
    if args.save == "True" or args.save == "true" or args.save == "true ":
        args.save = True

    if args.v_model == "False" or args.v_model == "false" or args.v_model == "false ":
        args.v_model = False
    if args.v_model == "True" or args.v_model == "true" or args.v_model == "true ":
        args.v_model = True

    if args.v_loss == "False" or args.v_loss == "false" or args.v_loss == "false ":
        args.v_loss = False
    if args.v_loss == "True" or args.v_loss == "true" or args.v_loss == "true ":
        args.v_loss = True

    if args.v_data == "False" or args.v_data == "false" or args.v_data == "false ":
        args.v_data = False
    if args.v_data == "True" or args.v_data == "true" or args.v_data == "true ":
        args.v_data = True
