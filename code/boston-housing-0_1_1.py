'''
Author: Lupos
Started: 08.09.2019
Lang: Phyton
Description: Prediction of boston housing market with lienar - regression.
version: 0.1.1

Dataset:
Housing Values in Suburbs of Boston

RM: average number of rooms per dwelling(Wohnung)
LSTAT: percentage of population considered lower status
PTRATIO: pupil-teacher ratio by town
MEDV: median value of owner-occupied homes in 10.000$
'''
import argparse
import csv
import os
from os import path
import multiprocessing as mp
import time
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import urllib
import operator

# TODO: fix train and test loss

# GLOBAL VARIABLES
global checker_dataset_exist  # gets set on true from get_Data()
checker_dataset_exist = False
# pool = multiprocessing.Pool(3) # set the pool(how many kernels) are used for multiprocessing
visualize_process = None  # gets later used from multiprocessing

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
    axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

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

class LinearRegression:
    def __init__(self, df):
        # weight/bias init
        self.w1 = 0
        self.bias = 1

        # how man epoch we train
        self.epochs = 30
        self.alpha = 0.03
        self.train_loss_history = []
        self.test_loss_history = []
        self.x_train_loose = []

        # split in target and data
        self.data_train = df[0]["RM"].tolist()
        self.target_train = df[0]["MEDV"].tolist()

        self.data_test = df[1]["RM"].tolist()
        self.target_test = df[1]["MEDV"].tolist()

        # misc
        self.evaluation_time = 0

    # training our model
    def train(self) -> None:

        while True:
            try:
                # get input for our model
                epochs = input("Please type the numbers of epoch you want to train: ")
                print(" ")
                epochs = int(epochs)
                self.epochs = epochs
                break
            except ValueError:
                print("Invalid Input!")

        start_time = time.time()
        for _ in range(self.epochs):
            train_loss_sum = 0
            test_loss_sum = 0
            for i in range(len(self.data_train)):
                # get our feature data from dataframe
                f1 = self.data_train[i]

                # our hypothesis/ what our model predicts
                pred_target = self.w1 * f1 + self.bias

                # update our weights
                self.bias = self.bias - (self.alpha * (pred_target - self.target_train[i]))
                self.w1 = self.w1 - (self.alpha * (pred_target - self.target_train[i]) * f1)

                # sums train loss
                train_loss = loss(pred_target, self.target_train[i])
                train_loss_sum += train_loss

                # test train loss
                if i < len(self.data_test):  # because test and train set have different sizes
                    f1 = self.data_test[i]
                    pred_target = self.w1 * f1 + self.bias
                    test_loss = loss(pred_target, self.target_test[i])
                    test_loss_sum += test_loss

                if args.fd == "full":
                    print("Epoch" + str(_) + " Example" + str(i) + ".Train loss: ", str(round(train_loss, 6)))  # prints loss for each example

            # save history of train loss for later use
            mean_loss_one_epoch_train = train_loss_sum / len(self.data_train)
            self.train_loss_history.append(mean_loss_one_epoch_train)
            self.x_train_loose.append(_)

            # save history of test loss for later use
            mean_loss_one_epoch_test = test_loss_sum / len(self.data_test)
            self.test_loss_history.append(mean_loss_one_epoch_test)

            # prints train loss
            if args.fd == "intermediate" or args.fd == "full":
                # when feedback=strong activate we want a little bit more space between the messages
                if args.fd == "full":
                    print(" ")
                    print("Epoch" + str(_) + " Mean-train loss: " + str(round(mean_loss_one_epoch_test, 6)))  # prints mean-loss of every Epoch
                    print(" ")
                else:
                    print("Epoch" + str(_) + " Mean-train loss: " +
                          str(mean_loss_one_epoch_test))  # prints mean-loss of every Epoch

            end_time = time.time()
            self.evaluation_time = end_time - start_time

    # saves weight and bias
    def save(self) -> None:
        filename = "linear_regression_housing_weights.csv"
        row = [["weight_bias"], [float(self.w1)], [float(self.bias)]]
        with open(filename, "w+", newline='') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(row)

        writeFile.close()

    # predicting with the model
    def predic(self) -> None:
        time.sleep(1)  # sleeps so that the function visualize()(which is seperate process through multiprocessing) has enough time to print the output correctly
        self.pred_target = 0
        print(" ")
        print("Prediction")
        print("------------------------------------")
        print("With this model you can predict how much a house is worth.")
        print("Based on RM short for the average number of rooms per dwelling(GER: Wohnung).")
        print(" ")
        # while true until valid input
        while True:
            try:
                # get input for our model
                print("Please enter the RM vaule. Values with the type of Int or float are only allowed.")
                print("If you want to quit type: 'quit'.")
                rm_input = input()
                if rm_input == "quit" or rm_input == "Quit":
                    if visualize_process.is_alive():
                        try:
                            visualize_process.terminate()
                        except Exception as e:
                            print("Error: ", str(e))
                    sys.exit(0)  # exit the script sucessful
                    break
                else:
                    rm_input = round(float(rm_input), 4)
                    self.pred_target = self.w1 * rm_input + self.bias  # predicting

                    print(" ")
                    print("The model predicted that a house with a RM value of: " + str(rm_input) + ".")
                    print("Is worth about: " + str(round(self.pred_target, 4)) + " in 10,000$(GER 10.000$).")
                    print(" ")
                    print("Please be noted that this value is a estimate. I am not liable responsibly.")
                    print("For more information about the copyright of this programm look at my Github repository: ")
                    print("github.com/LuposX/BostonHousingPrediction")
                    if visualize_process.is_alive():
                        try:
                            visualize_process.terminate()
                        except Exception as e:
                            print("Error: ", str(e))
                    sys.exit(0)  # exit the script sucessful
                    break
            except ValueError:
                print("Invalid Input!")

    # a getter for the viszulation function
    def getter_viszualtion(self) -> list:
        return self.w1, self.bias, self.train_loss_history, self.test_loss_history, self.evaluation_time, self.data_train, self.target_train, self.x_train_loose


if __name__ == "__main__":
    # create our parser for commands from command line
    parser = argparse.ArgumentParser(description="This is a program which creates a prediction model for the boston "
                                                 "housing dataset.")

    # available options for the command line use
    parser.add_argument("model", help="Choose which model you want to use for prediction.",
                        type=str, choices=["linear_regression", "polynomial_regression"])
    parser.add_argument("--infile", help="If file specified model will load weights from it."
                                       "Else it will normally train.(default: no file loaded)"
                        , metavar="FILE", type=str) # type=argparse.FileType('r', encoding='UTF-8')
    # implemented
    parser.add_argument("--v_data", metavar="VISUALIZE_DATA",
                        help="Set it to True if you want to get a visualization of the data.(default: %(default)s)",
                        type=bool, default=False)
    # implemented
    parser.add_argument("--v_loss", metavar="VISUALIZE_LOSS",
                        help="Set it to True if you want to get a visualization of the loss.(default: %(default)s)",
                        type=bool, default=False)
    # implemented
    parser.add_argument("--v_model", metavar="VISUALIZE_MODEL",
                        help="Set it to True if you want to get a visualization of the model.(default: %(default)s)",
                        type=bool, default=False)
    parser.add_argument("--fd",  metavar="FEEDBACK",
                        help="Set how much feedback you want.(Choices: %(choices)s)",
                        type=str, choices=["full", "intermediate", "weak"], default="immediate")
    # implemented
    parser.add_argument("--save", metavar="SAVE_MODEL", help="Set it to True if you want to save the model after training.(default: %(default)s)",
                        type=bool, default=False)

    parser.add_argument("--h_features", metavar="HELP_FEATURES",
                        help="Set it to True if you want to print out the meaning of the features in the dataset.(default: %(default)s)",
                        type=bool, default=False)

    # parse the arguments
    args = parser.parse_args()

    # check if the dataset exist
    if not checker_dataset_exist:
       download_dataset()
       df_data = get_Data()
    else:
        df_data = get_Data()

    # check arguments programm got started with
    if args.model == "linear_regression" and not args.h_features:
        print(" ")
        print("Linear-regression")
        print("--------------------------------------")

        model = LinearRegression(preproc_data(df_data))  # create our model

        if not args.infile:
            model.train()  # train our model
        else:
            try:
                filename = str(args.infile)
                file_list = []
                with open(filename, "r") as infile:
                    for line in chomped_lines(infile):
                        file_list.append(line)
            except FileNotFoundError as e:
                print("Errot file not found: ", str(e))
                if visualize_process.is_alive():
                    visualize_process.terminate()
                sys.exit(1)  # exit the script sucessful

            try:
                model.w1 = float(file_list[1])
                model.bias = float(file_list[2])
            except ValueError as e:
                print(str(e))


        # if save parameter is true model gets saved
        if args.save and not args.infile:
            model.save()

        # output_mp = mp.Queue()# Define an output queue. for multiprocessing
        # Setup a  processes that we want to run
        # processes = [mp.Process(target=rand_string, args=(5, x, output)) for x in range(4)] # if we later want multiples processe
        # args, df_data, self.w1, self.bias, self.train_loss_history, self.test_loss_history, self.evaluation_time, self.data_train, self.target_train
        random.seed(123)  # needed to fix some issued with multiprocessing
        list_process_arg = model.getter_viszualtion()
        visualize_process = mp.Process(target=visualize, args=(args, df_data, list_process_arg))  # use "args" if arguments are needed
        visualize_process.start()
        # model.visualize(args, df_data)  # visualize our model
        model.predic()  # make preictions with the model

    elif args.model == "polynomial_regression":
        print(" ")
        print("Polynomial-regression")
        print("--------------------------------------")
        print("This model doesn't exist yet. And is currently under Development.")

    elif args.h_features:
        print(" ")
        print("Features and their meaning")
        print("-----------------------------------------")
        print("RM: average number of rooms per dwelling(GER: Wohnungen).")
        print("LSTAT: percentage of population considered lower status.")
        print("PTRATIO: pupil-teacher ratio by town")
        print(" ")
        print("Target and it's meaning")
        print("-----------------------------------------")
        print("MEDV: median value of owner-occupied homes in 10,000$(GER: 10.000$).")
