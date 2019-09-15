import urllib
import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def loss(pred_target: float, real_traget: float) -> float:
    return float(np.sqrt((pred_target - real_traget) ** 2))


def get_Data() -> object:
    try:
        if path.isfile("boston_housing.csv"):
            df = pd.read_csv("boston_housing.csv")
            checker_dataset_exist = True
            return df
    except:
        try:
            if path.isfile("housing.csv"):
                df = pd.read_csv("housing.csv")
                checker_dataset_exist = True
                return df
        except FileNotFoundError:
            print("oops, file doesn't exist")
            checker_dataset_exist = False


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


def preproc_data(df: object) -> None:
    # regularization and rounding on the 4th decimal point
    df[["MEDV"]] = round(df[["MEDV"]] / 10000, 4)

    # shuffling data
    df = df.sample(frac=1).reset_index(drop=True)

    # split in training and test data
    df_train = df[:380]
    df_test = df[381:]

    return df_train, df_test


# visualize our model. the function visualize() is not in the class model so that we can use multiprocessing.
# args, df_data, self.w1, self.bias, self.train_loss_history, self.test_loss_history, self.evaluation_time, self.data_train, self.target_train
def visualize(args, df_data, parameter_list: list) -> None:
    # unzip the argument list gotten from model.getter_viszulation()
    w1 = parameter_list[0]
    bias = parameter_list[1]
    train_loss_history = parameter_list[2]
    test_loss_history = parameter_list[3]
    evaluation_time = parameter_list[4]
    data_train = parameter_list[5]
    target_train = parameter_list[6]
    x_train_loose = parameter_list[7]

    # prints Mean loss of last epoch
    if not args.infile:
        print(" ")
        print("Mean-train loss of last epoch: ", str(round(train_loss_history[-1], 6)))
        print("Mean-test loss of last epoch:  ", str(round(test_loss_history[-1], 6)))
        print("Time needed for training:      ", str(round(evaluation_time, 4)) + "s.")

    # communication is key
    if (args.fd == "intermediate" or args.fd == "full") and not args.infile:
        print(" ")
        print("Value of W1 after training:   ", w1)
        print("Value of Bias after training: ", bias)
        print(" ")
    elif args.infile:
        print(" ")
        print("Value of W1:   ", w1)
        print("Value of Bias: ", bias)
        print(" ")

    if args.v_data:
        visualize_Data(df_data)

    # get points for line
    X = []
    Y = []
    for i in range(3, 11):
        X.append(i)
        Y.append(w1 * i + bias)


    # plot our descion border and datapoints
    if args.v_model:
        sns.set_style("darkgrid")
        plt.figure(figsize=(9, 6))
        plt.title("Decision Border and Data-points")
        plt.xlabel("Average number of rooms per dwelling(Wohnung)")
        plt.ylabel("Median value of owner-occupied homes in 1000$")
        sns.lineplot(x=X, y=Y)
        sns.scatterplot(x=data_train, y=target_train, color="green")

    # convert our loss arrays into a dataframe from pandas
    # print("x_train: ", str(len(x_train_loose)))
    # print("train_loss: ", str(len(train_loss_history)))
    # print("test_loss_history: ", str(len(test_loss_history)))
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

    if args.v_loss or args.v_model or args.v_data:
        plt.show()