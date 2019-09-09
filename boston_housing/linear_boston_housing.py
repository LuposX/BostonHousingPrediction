'''
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
import argparse
import csv

# TODO: fix train and test loss
# TODO: add verify  verify a single preictions
# TODO: programm should run like console programm with different command

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def loss(pred_target: float, real_traget: float) -> float:
    return float(np.sqrt((pred_target - real_traget) ** 2))


def get_Data() -> object:
    df = pd.read_csv("datasets/boston_housing.csv")
    return df


# visualize data
def visualize_Data(df: object) -> None:
    """
    :type df: Dataframe from pandas
    """
    sns.set_style("darkgrid")
    plt.figure(figsize=(9, 6))
    sns.distplot(df["MEDV"], bins=30)
    plt.show()


def is_non_zero_file() -> object:
    return (os.path.isfile("housing.csv") and os.path.getsize("housing.csv") > 0) or (
            os.path.isfile("boston_housing.csv") and os.path.getsize("boston_housing.csv") > 0)


def download_dataset() -> None:
    try:
        file = open("boston_housing.csv", "w+")  # the plus asgins when not exist we create it
        data = urllib.request.urlopen(
            "https://raw.githubusercontent.com/udacity/machine-learning/master/projects/boston_housing/housing.csv").read().decode(
            'utf-8')
        file.write(data)
        file.close()
    except:
        try:
            file = open("boston_housing.csv", "w+")  # the plus asgins when not exist we create it
            data = urllib.request.urlopen(
                "https://raw.githubusercontent.com/LuposX/BostonHousingPrediction/master/dataset/boston_housing.csv").read().decode(
                'utf-8')
            file.write(data)
            file.close()
        except Exception as e:
            print("Error: ", str(e))


class LinearRegression:
    def __init__(self):
        # weight/bias init
        self.w1 = 0
        self.bias = 1

        # how man epoch we train
        self.epochs = 30
        self.alpha = 0.03
        self.train_loss_history = []
        self.test_loss_history = []
        self.x_train_loose = []

        self.data_train = 0
        self.target_train = 0
        self.data_test = 0
        self.target_test = 0

    def preproc_data(self, df: object) -> None:
        # regularization and rounding on the 4th decimal point
        df[["MEDV"]] = round(df[["MEDV"]] / 1000, 4)

        # shuffling data
        df = df.sample(frac=1).reset_index(drop=True)

        # split in training and test data
        df_train = df[:380]
        df_test = df[381:]

        # split in target and data
        self.data_train = df_train["RM"].tolist()
        self.target_train = df_train["MEDV"].tolist()

        self.data_test = df_test["RM"].tolist()
        self.target_test = df_test["MEDV"].tolist()

    def train(self) -> None:
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
                    test_loss_sum += train_loss

            # save history of train loss for later use
            mean_loss_one_epoch_train = train_loss_sum / len(self.data_train)
            self.train_loss_history.append(mean_loss_one_epoch_train)
            self.x_train_loose.append(_)

            # save history of test loss for later use
            mean_loss_one_epoch_test = test_loss_sum / len(self.data_test)
            self.test_loss_history.append(mean_loss_one_epoch_test)

            # prints train loss
            print("Train loss: ", train_loss)

        # saves weight and bias
        def safe():
            filename = "linear_regression_housing_weights.csv"
            row = [["weight_bias"], [float(self.w1)], [float(self.bias)]]
            with open(filename, "w+", newline='') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows(row)

            writeFile.close()

    def visualize(self) -> None:
        # communication is key
        print(" ")
        print("Bias: ", self.bias)
        print("W1: ", self.w1)

        # get points for line
        X = []
        Y = []
        for i in range(3, 11):
            X.append(i)
            Y.append(self.w1 * i + self.bias)

        # plot our descion border and datapoints
        sns.set_style("darkgrid")
        plt.figure(figsize=(9, 6))
        plt.title("Decision Border and Data-points")
        plt.xlabel("Average number of rooms per dwelling(Wohnung)")
        plt.ylabel("Median value of owner-occupied homes in 1000$")
        sns.lineplot(x=X, y=Y)
        sns.scatterplot(x=self.data_train, y=self.target_train, color="green")

        # convert our loss arrays into a dataframe from npandas
        data = {"x": self.x_train_loose, "train": self.train_loss_history, "test": self.test_loss_history}
        data = pd.DataFrame(data, columns=["x", "train", "test"])

        # plot loss over time
        sns.set_style("darkgrid")
        plt.figure(figsize=(12, 6))
        plt.xlabel("Time In Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Time")
        sns.lineplot(x="x", y="train", data=data, label="train", color="orange")
        sns.lineplot(x="x", y="test", data=data, label="test", color="Violet")
        plt.show()


if __name__ == "__main__":
    model = LinearRegression()
    model.preproc_data(get_Data())
    model.train()
    model.visualize()

    # create our parser for commands from command line
    parser = argparse.ArgumentParser(description="This is a program which creates a prediction model for the boston "
                                                 "housing dataset.")

    # available options for the command line use
    parser.add_argument("Model", help="Choose which model you want to use for prediction.",
                        type=str, choices=["linear_regression", "polynomial_regression"])
    parser.add_argument("--infile", help="If file specified model will load weights from it."
                                       "Else it will normally train.(default: no file loaded)"
                        , metavar="FILE", type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument("--v_data", metavar="VISUALIZE_DATA",
                        help="Set if you want to get a visualization of the data.(default: %(default)s)",
                        type=bool, default=False)
    parser.add_argument("--v_loss", metavar="VISUALIZE_LOSS",
                        help="Set if you want to get a visualization of the loss.(default: %(default)s)",
                        type=bool, default=False)
    parser.add_argument("--v_model", metavar="VISUALIZE_MODEL",
                        help="Set if you want to get a visualization of the model.(default: %(default)s)",
                        type=bool, default=False)
    parser.add_argument("--fd", help="Set how much feedback you want.(default: %(default)s)",
                        type=str, choices=["full", "immediate", "weak"], default="immediate")
    parser.add_argument("--save", metavar="SAVE_MODEL", help="Set if you want to save the model after training.(default: %(default)s)",
                        type=bool, default=False)

    args = parser.parse_args()

    if args.Model == "linear_regression":
        print("linear")
    elif args.Model == "polynomial_regression":
        print("polynomial")

    # check if the dataset exist
    if not is_non_zero_file():
       download_dataset()

