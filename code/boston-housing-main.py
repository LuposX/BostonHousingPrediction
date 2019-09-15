'''
Author: Lupos
Started: 08.09.2019
Lang: Phyton
Description: Prediction of boston housing market with lienar - regression.
version: 0.1.2

Dataset:
Housing Values in Suburbs of Boston

RM: average number of rooms per dwelling(Wohnung)
LSTAT: percentage of population considered lower status
PTRATIO: pupil-teacher ratio by town
MEDV: median value of owner-occupied homes in 10.000$
'''
import argparse
import multiprocessing as mp
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import urllib
import operator

# My Files that get imported
from linear_regression_libary import LinearRegression
from misc_libary import *

# TODO: fix train and test loss

# GLOBAL VARIABLES
global checker_dataset_exist  # gets set on true from get_Data()
checker_dataset_exist = False
# pool = multiprocessing.Pool(3) # set the pool(how many kernels) are used for multiprocessing
visualize_process = None  # gets later used from multiprocessing

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

        model = LinearRegression(preproc_data(df_data), args)  # create our model

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
        model.predic(visualize_process)  # make preictions with the model

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
