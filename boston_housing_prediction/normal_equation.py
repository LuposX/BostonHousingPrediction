import pandas as pd
import numpy as np
import time
import sys

class NormalEquation():
    def __init__(self, df_data, args) -> None:
        # START: preprocessing data
        # ---------------------------------------
        # defining df
        df = df_data

        # split target and data
        df_data = df[["RM", "LSTAT", "PTRATIO"]]
        self.df_target = df[["MEDV"]]

        # adding features such as RM ** 2 or RM ** 3
        name_list = ["RM", "LSTAT", "PTRATIO"]
        for i in name_list:
            for _ in range(2, 4):
                df_data[str(i) + str(_)] = df_data[i] ** _

        # define order od dataframe
        self.cols = ["RM", "RM2", "RM3", "LSTAT", "LSTAT2", "LSTAT3", "PTRATIO", "PTRATIO2", "PTRATIO3"]

        df_data = df_data[self.cols]

        # inserting bias
        df_data["bias"] = np.ones(len(df_data[self.cols[0]]))

        self.df_data = df_data
        # END: preprocessing data
        # ---------------------------------------

        # init weights
        self.weight = []
        self.bias = 1

        # misc
        self.duration_time = 0
        self.args = args


    def train(self):
        # for caluclating how long it took to calculate
        start_time = time.time()

        # calculating our weights with the formula:
        # theta = (XT * X) ^-1  * (XT * y)
        self.weights = np.dot(np.linalg.inv(np.dot(self.df_data.transpose(), self.df_data)), np.dot(self.df_data.transpose(), self.df_target))

        # removes scientific notation e.g. 2.2331 e-2
        np.set_printoptions(suppress=True)
        cols = self.cols
        cols.append("Bias")
        # printing the weights
        if self.args.fd == "intermediate" or self.args.fd == "debug" or self.args.fd == "full":
            for i, co in enumerate(cols):
                print(str(co + ": "), str(self.weights[i]))

        np.set_printoptions(suppress=False)
        # end time
        end_time = time.time()

        # calculating the needed time
        self.duration_time = start_time - end_time

    # our hypothesis/ what our model predicts
    def hypothesis(self, weights, f1, f2, f3, bias):
        pred = weights[0] * f1 + weights[1] * f1 ** 2 + weights[2] * f1 ** 3 + \
               weights[3] * f2 + weights[4] * f2 ** 2 + weights[5] * f2 ** 3 + \
               weights[6] * f3 + weights[7] * f3 ** 2 + weights[8] * f3 ** 3 + \
               weights[9] * bias

        return pred

    # predicting with the model
    # visualize_process
    def predic(self, visualize_process) -> None:
        time.sleep(1)  # sleeps so that the function visualize()(which is seperate process through multiprocessing) has enough time to print the output correctly
        self.pred_target = 0
        print(" ")
        print("Prediction")
        print("------------------------------------")
        print("With this model you can predict how much a house is worth.")
        print(" ")
        # while true until valid input
        while True:
            try:
                print('If you want to quit type: "quit".')
                print('Only Values with the type of "int" or "float" are allowed.')
                print("Type the Values in the following order: ")
                print("1.RM 2.LSTAT 3.PTRATIO")
                input_list = []
                default_values = [6.24, 12.94, 18.52]  # those are the default values when field is left empty. default values corrospond to mean values of feature
                for i in range(0, 3, 1):
                    # exits while loop when right inputs got inserted
                    while True:
                        input_var = input() or default_values[i]

                        if input_var == "quit" or input_var == "Quit":
                            if visualize_process.is_alive():
                                try:
                                    visualize_process.terminate()
                                except Exception as e:
                                    print("Error: ", str(e))
                            print(" ")
                            print("Please be noted that this value is a estimate. I am not liable responsibly.")
                            print(
                                "For more information about the copyright of this programm look at my Github repository: ")
                            print("github.com/LuposX/BostonHousingPrediction")
                            sys.exit(0)  # exit the script sucessful
                            break

                        try:
                            input_var = float(input_var)
                            if input_var < 0:
                                print("Please don't enter negative numbers :)")
                            else:
                                break

                        except ValueError:
                            print("Invalid Input :/")

                    input_list.append(input_var)

            except Exception as e:
                print(str(e))

            try:
                print(" ")

                # typecasting our inputs and rounding them
                rm_input = round(float(input_list[0]), 4)  # or 5 is the default value
                lstat_input = round(float(input_list[1]), 4)
                ptratio_input = round(float(input_list[2]), 4)

                # predicting
                predic_target = self.hypothesis(self.weights, rm_input, lstat_input, ptratio_input, self.bias)

                # check if predicted output is negative
                if predic_target < 0:
                    print(" ")
                    print("-----------------------------------------------------------------------------")
                    print("The model predicted that a house with the values: ")
                    print("RM :" + str(rm_input))
                    print("LSTAT :" + str(lstat_input))
                    print("PTRATIO :" + str(ptratio_input))
                    print("Warning: the input values doesn't correspond to a real house.")
                    print("-----------------------------------------------------------------------------")
                    print(" ")
                else:
                    print(" ")
                    print("-----------------------------------------------------------------------------")
                    print("The model predicted that a house with the values: ")
                    print("RM :" + str(rm_input))
                    print("LSTAT :" + str(lstat_input))
                    print("PTRATIO :" + str(ptratio_input))
                    print("Is worth about: " + str(predic_target) + " in 10,000$(GER 10.000$).")
                    print("-----------------------------------------------------------------------------")
                    print(" ")

            except Exception as e:
                 print("Something went wrong: ", str(e))

    # a getter for the viszulation function
    def getter_viszualtion(self):
        return self.weights, self.duration_time

