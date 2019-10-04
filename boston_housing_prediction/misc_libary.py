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

# global variables
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


# saves weight and bias
def save(weights, args) -> None:
    if args.model == "polynomial_regression":
        filename = "polynomial_regression_housing_weights.txt"

    elif args.model == "linear_regression":
        filename = "linear_regression_housing_weights.txt"

    elif args.model == "normal_equation":
        filename = "normal_equation_housing_weights.txt"

    with open(filename, "w+", newline='') as writeFile:
        for i in weights:
            writeFile.write(str(i) + "\n")

    writeFile.close()
