'''
Author: Lupos
Started: 08.09.2019
Lang: Phyton
Description: Prediction of boston housing market with regression.

Dataset:
Housing Values in Suburbs of Boston

RM: average number of rooms per dwelling(Wohnung)
LSTAT: percentage of population considered lower status
PTRATIO: pupil-teacher ratio by town
MEDV: median value of owner-occupied homes
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("datasets/boston_housing.csv")
    
    # # visualize data
    # sns.set_style("darkgrid")
    # plt.figure(figsize=(9, 6))
    # sns.distplot(df["MEDV"], bins=30)
    # plt.show()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def loss(pred_target, real_traget):
        return np.sqrt((pred_target - real_traget) ** 2)

    # split in training and test data
    df_train = df[:380]
    df_test = df[381:]

    # split in target and data
    data_train = df_train[["RM", "LSTAT", "PTRATIO"]]
    target_train = df_train["MEDV"]

    data_test = df_test[["RM", "LSTAT", "PTRATIO"]]
    target_test = df_test["MEDV"]

    # weight/bias init
    w1 = np.random.uniform(low=-3, high=3)
    w2 = np.random.uniform(low=-3, high=3)
    w3 = np.random.uniform(low=-3, high=3)
    bias = 1

    # how man epoch we train
    epochs = 20
    alpha = 0.03
    for i in range(epochs):
        # get our feature data from dataframe
        f1 = data_train[data_train.index == i]["RM"]
        f2 = data_train[data_train.index == i]["LSTAT"]
        f3 = data_train[data_train.index == i]["PTRATIO"]

        # our hypothesis/ what our model predicts
        pred_target = (w1 * f1) + (w2 * f2) + (w3 * f3) + bias

        # update our weights
        bias = bias - (alpha * (pred_target - target_train[i]))
        w1 = w1 - (alpha * (pred_target - target_train[i]) * (f1+f2+f3))
        w2 = w1 - (alpha * (pred_target - target_train[i]) * (f1 + f2 + f3))
        w3 = w1 - (alpha * (pred_target - target_train[i]) * (f1 + f2 + f3))

        print("target ", target_train[i])
        print("pred ", pred_target)
        print("bias ", bias - (alpha * (pred_target - target_train[i])))
        # print("Current loss: ", loss(pred_target, target_train[i]))


