import argparse
import multiprocessing as mp
import random
import sys

# My Files that get imported
# from boston_housing_prediction.linear_regression_libary import LinearRegression
# from boston_housing_prediction.polynomial_regression_libary import PolynomialRegression
# from boston_housing_prediction.misc_libary import *
from future.types import no
from linear_regression_libary import LinearRegression
from polynomial_regression_libary import PolynomialRegression
from misc_libary import *
from normal_equation import *

# TODO: fix train and test loss

# GLOBAL VARIABLES
visualize_process = None  # gets later used from multiprocessing


def main():
    # create our parser for commands from command line
    parser = argparse.ArgumentParser(description="This is a program which creates a prediction model for the boston "
                                                 "housing dataset.")

    # available options for the command line use
    parser.add_argument("model", help="Choose which model you want to use for prediction.",
                        type=str, choices=["linear_regression", "polynomial_regression", "normal_equation"])
    parser.add_argument("--infile", help="If file specified model will load weights from it."
                                         "Else it will normally train.(default: no file loaded)"
                        , metavar="FILE", type=str)  # type=argparse.FileType('r', encoding='UTF-8')
    # implemented
    parser.add_argument("--v_data", metavar="VISUALIZE_DATA",
                        help="Set it to True if you want to get a visualization of the data.(default: %(default)s)",
                        type=str, default="False")
    # implemented
    parser.add_argument("--v_loss", metavar="VISUALIZE_LOSS",
                        help="Set it to True if you want to get a visualization of the loss.(default: %(default)s)",
                        type=str, default="False")
    # implemented
    parser.add_argument("--v_model", metavar="VISUALIZE_MODEL",
                        help="Set it to True if you want to get a visualization of the model.(default: %(default)s)",
                        type=str, default="False")
    parser.add_argument("--fd", metavar="FEEDBACK",
                        help="Set how much feedback you want.(Choices: %(choices)s)",
                        type=str, choices=["full", "intermediate", "weak", "debug"], default="intermediate")
    # implemented
    parser.add_argument("--save", metavar="SAVE_MODEL",
                        help="Set it to True if you want to save the model after training.(default: %(default)s)",
                        type=str, default="False")

    parser.add_argument("--h_features", metavar="HELP_FEATURES",
                        help="Set it to True if you want to print out the meaning of the features in the dataset.(default: %(default)s)",
                        type=str, default="False")

    parser.add_argument("--predict_on",
                        help="Set it to False if you dont want to predict after training.(default: %(default)s)",
                        type=str, default="True")

    # parse the arguments
    args = parser.parse_args()

    # convert our arguments from strings into booleans
    parse_bool_args(args)

    # check if the dataset exist
    if not is_non_zero_file():
        download_dataset()
        df_data = get_Data()
    else:
        df_data = get_Data()

    # df variables
    df_arg_list = preproc_data(df_data, args)
    df_args = df_arg_list[0:2]
    args_normalization = df_arg_list[2:]

    # check arguments programm got started with
    if args.model == "linear_regression" and not args.h_features:
        print(" ")
        print("Linear-regression")
        print("--------------------------------------")

        model_line = LinearRegression(df_args, args)  # create our model

        if not args.infile:
            model_line.train()  # train our model
        else:
            try:
                filename = str(args.infile)
                file_list = []
                with open(filename, "r") as infile:
                    for line in chomped_lines(infile):
                        file_list.append(line)
            except FileNotFoundError as e:
                print("Error file not found: ", str(e))
                if visualize_process.is_alive():
                    visualize_process.terminate()
                sys.exit(1)  # exit the script sucessful

            try:
                model_line.w1 = float(file_list[0])
                model_line.bias = float(file_list[1])
            except ValueError as e:
                print(str(e))

        # if save parameter is true model gets saved
        if args.save and not args.infile:
            model_line.save()

        # START: visualisation
        # ------------------------
        random.seed(123)  # needed to fix some issued with multiprocessing
        list_process_arg = model_line.getter_viszualtion()

        # visualizing is in a new process
        visualize_process = mp.Process(target=visualize,
                                       args=(args, df_args, list_process_arg))  # use "args" if arguments are needed
        visualize_process.start()
        # END: visualisation
        # ------------------------

        if args.predict_on:
            model_line.predic(visualize_process, args_normalization)  # make preictions with the model

    elif args.model == "polynomial_regression":
        print(" ")
        print("Polynomial-regression with GradientDescent")
        print("--------------------------------------")

        model_poly = PolynomialRegression(df_args, args)  # create our model

        if not args.infile:
            model_poly.train()  # train our model
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
                model_poly.weights = file_list
            except ValueError as e:
                print(str(e))

        # if save parameter is true model gets saved
        if args.save and not args.infile:
            model_poly.save()

        # START: visualisation
        # ------------------------
        random.seed(123)  # needed to fix some issued with multiprocessing
        list_process_arg = model_poly.getter_viszualtion()

        # visualizing is in a new process
        visualize_process = mp.Process(target=visualize,
                                       args=(args, df_args, list_process_arg))  # use "args" if arguments are needed
        visualize_process.start()
        # END: visualisation
        # ------------------------

        if args.predict_on:
            model_poly.predic(visualize_process, args_normalization)  # make preictions with the model

    elif args.model == "normal_equation":
        print(" ")
        print("Polynomial-regression with NormalEquation")
        print("--------------------------------------")

        model_norm = NormalEquation(df_data, args)
        model_norm.train()

        # START: visualisation
        # ------------------------
        random.seed(123)  # needed to fix some issued with multiprocessing
        list_process_arg = model_norm.getter_viszualtion()

        # visualizing is in a new process
        visualize_process = mp.Process(target=visualize,
                                       args=(args, df_args, list_process_arg))  # use "args" if arguments are needed
        visualize_process.start()
        # END: visualisation
        # ------------------------

        if args.predict_on:
            model_norm.predic(visualize_process)  # make preictions with the model

    # print what the feature shortcuts means
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
        print("MEDV: median value of owner-occupied homes in 100,000$(GER: 100.000$).")


if __name__ == "__main__":
    main()
