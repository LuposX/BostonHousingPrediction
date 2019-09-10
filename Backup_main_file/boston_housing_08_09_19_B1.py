'''
BACKUP from: 09.09.2019
Author: Lupos
Started: 08.09.2019
Lang: Phyton
Description: Prediction of boston housing market with lienar - regression.

Dataset:
Housing Values in Suburbs of Boston

RM: average number of rooms per dwelling(Wohnung)
LSTAT: percentage of population considered lower status
PTRATIO: pupil-teacher ratio by town
MEDV: median value of owner-occupied homes in 1000$
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# TODO: fix train and test loss
# TODO: add verify  verify a single preictions
# TODO: programm should run like console programm with different command

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss(pred_target, real_traget):
    return float(np.sqrt((pred_target - real_traget) ** 2))


def get_Data():
    df = pd.read_csv("datasets/boston_housing.csv")
    return df


# visualize data
def visualize_Data():
    sns.set_style("darkgrid")
    plt.figure(figsize=(9, 6))
    sns.distplot(df["MEDV"], bins=30)
    plt.show()

if __name__ == "__main__":
    # regularization and rounding on the 4th decimal point
    df[["MEDV"]] = round(df[["MEDV"]] / 1000, 4)

    # shuffling data
    df = df.sample(frac=1).reset_index(drop=True)

    # split in training and test data
    df_train = df[:380]
    df_test = df[381:]

    # split in target and data
    data_train = df_train["RM"].tolist()
    target_train = df_train["MEDV"].tolist()

    data_test = df_test["RM"].tolist()
    target_test = df_test["MEDV"].tolist()

    # weight/bias init
    w1 = 0
    bias = 1

    # how man epoch we train
    epochs = 30
    alpha = 0.03
    train_loss_history = []
    test_loss_history = []
    x_train_loose = []
    curr_epochs = 0  # curren epoch io nwhich epoch we are currently
    for _ in range(epochs):
        train_loss_sum = 0
        test_loss_sum = 0
        for i in range(len(data_train)):
            # get our feature data from dataframe
            f1 = data_train[i]

            # our hypothesis/ what our model predicts
            pred_target = w1 * f1 + bias

            # update our weights
            bias = bias - (alpha * (pred_target - target_train[i]))
            w1 = w1 - (alpha * (pred_target - target_train[i]) * f1)

            # sums train loss
            train_loss = loss(pred_target, target_train[i])
            train_loss_sum += train_loss

            # test train loss
            if i < len(data_test):  # because test and train set have different sizes
                f1 = data_test[i]
                pred_target = w1 * f1 + bias
                test_loss = loss(pred_target, target_test[i])
                test_loss_sum += train_loss

        # save history of train loss for later use
        mean_loss_one_epoch_train = train_loss_sum / len(data_train)
        train_loss_history.append(mean_loss_one_epoch_train)
        x_train_loose.append(_)

        # save history of test loss for later use
        mean_loss_one_epoch_test = test_loss_sum / len(data_test)
        test_loss_history.append(mean_loss_one_epoch_test)

        # prints train loss
        print("Train loss: ", train_loss)

    # # testing
    # test_loss_history = []
    # x_test_loose = []
    # for i in range(len(df_test)):
    #     # get our feature data from dataframe
    #     f1 = data_test[i]
    #
    #     # our hypothesis/ what our model predicts
    #     pred_target = w1 * f1 + bias
    #
    #     # save history of loss for later use
    #     test_loss = loss(pred_target, target_train[i])
    #     test_loss_history.append(test_loss)
    #     x_test_loose.append(i)
    #     # print("Train loss: ", test_loss_history)

    # communication is key
    print(" ")
    print("Bias: ", bias)
    print("W1: ", w1)

    # get points for line
    X = []
    Y = []
    for i in range(3, 11):
        X.append(i)
        Y.append(w1 * i + bias)

    # plot our descion border and datapoints
    sns.set_style("darkgrid")
    plt.figure(figsize=(9, 6))
    plt.title("Decision Border and Data-points")
    plt.xlabel("Average number of rooms per dwelling(Wohnung)")
    plt.ylabel("Median value of owner-occupied homes in 1000$")
    sns.lineplot(x=X, y=Y)
    sns.scatterplot(x=data_train, y=target_train, color="green")

    # convert our loss arrays into a dataframe from npandas
    data = {"x": x_train_loose, "train": train_loss_history, "test": test_loss_history}
    df = pd.DataFrame(data, columns=["x", "train", "test"])

    # plot loss over time
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 6))
    plt.xlabel("Time In Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Time")
    sns.lineplot(x="x", y="train", data=data, label="train", color="orange")
    sns.lineplot(x="x", y="test", data=data, label="test", color="Violet")
    plt.show()
