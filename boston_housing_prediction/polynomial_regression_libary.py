import time
from misc_libary import loss
import sys

class PolynomialRegression:
    def __init__(self, df, args):
        # weight/bias init
        self.weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.bias = 1

        self.epochs = 30   # how man epoch we train
        self.alpha = 0.003  # learning rate

        # initiate variables to visualize loss history
        self.train_loss_history = []
        self.test_loss_history = []
        self.x_train_loose = []

        # split in target and data
        self.data_train = df[0].iloc[:,  df[0].columns != "MEDV"].reset_index(drop=True)    # the ":" stands for every element in there
        self.data_test = df[1].iloc[:,  df[1].columns != "MEDV"].reset_index(drop=True)
        self.target_train = df[0]["MEDV"].tolist()
        self.target_test = df[1]["MEDV"].tolist()

        # misc
        self.evaluation_time = 0
        self.args = args

    # our hypothesis/ what our model predicts
    def hypothesis(self, weights, f1, f2, f3, bias):
        pred = weights[0] * f1 + weights[1] * f1 ** 2 + weights[2] * f1 ** 3 + weights[3] * f2 + weights[4] * f2 ** 2 + \
                          weights[5] * f2 ** 3 + weights[6] * f3 + weights[7] * f3 ** 2 + weights[8] * f3 ** 3 + weights[9] * bias
        return pred

    # training our model
    def train(self) -> None:
        # exits while loop when right inputs got inserted
        while True:
            try:
                # get input for our model
                epochs = input("Please type the numbers of epoch you want to train: ")
                print(" ")
                epochs = int(epochs)
                if epochs > 0:
                    self.epochs = epochs
                    break
                print("Please don't input negative numbers :)")
            except ValueError:
                print("Invalid Input!")

        start_time = time.time()   # start timer. To later calculate time needed to train the model
        for _ in range(self.epochs):
            train_loss_sum = 0
            test_loss_sum = 0

            for i in range(len(self.target_train)):
                # our features and their data
                f1 = self.data_train["RM"][i]
                f2 = self.data_train["LSTAT"][i]
                f3 = self.data_train["PTRATIO"][i]

                # our hypothesis/ what our model predicts
                pred_target = self.hypothesis(self.weights, f1, f2, f3, self.bias)

                # calculate the error(How far away was our prediction from the real value)
                error = (pred_target - self.target_train[i])

                # training our weights
                self.weights[0] = self.weights[0] - self.alpha * (error * f1)
                self.weights[1] = self.weights[1] - self.alpha * (2 * error * f1)
                self.weights[2] = self.weights[2] - self.alpha * (3 * error * f2 ** 2)
                self.weights[3] = self.weights[3] - self.alpha * (error * f2)
                self.weights[4] = self.weights[4] - self.alpha * (2 * error * f2)
                self.weights[5] = self.weights[5] - self.alpha * (3 * error * f2 ** 2)
                self.weights[6] = self.weights[6] - self.alpha * (error * f3)
                self.weights[7] = self.weights[7] - self.alpha * (2 * error * f3)
                self.weights[8] = self.weights[8] - self.alpha * (3 * error * f3 ** 2)
                self.weights[9] = self.weights[9] - self.alpha * (error * self.bias)

                # sums train loss
                train_loss = loss(pred_target, self.target_train[i])
                train_loss_sum += train_loss

                # outputs for debug mode
                if self.args.fd == "debug":
                    print(" ")
                    print("example: ", str(i))
                    print("----------------------")
                    print("Weight 1: ", str(self.weights[0]))
                    print("Weight 1 change: ", str(self.alpha * (error * f1)))
                    print("Weight 1 feature: ", str(f1))
                    print("Error: ", str(error))
                    print("----------------------")
                    print("Weight 6: ", str(self.weights[5]))
                    print("Weight 6 change: ", str(self.alpha * (3 * error * f2 ** 2)))
                    print("Weight 6 feature: ", str(f2))
                    print("Error: ", str(error))
                    print(" ")

                # test train loss
                if i < len(self.target_test):  # because test and train set have different sizes
                    # our features and their data
                    f1 = self.data_test["RM"][i]
                    f2 = self.data_test["LSTAT"][i]
                    f3 = self.data_test["PTRATIO"][i]

                    # predict test house prices
                    pred_target_test = self.hypothesis(self.weights, f1, f2, f3, self.bias)

                    # evaluate with loss
                    test_loss = loss(pred_target_test, self.target_test[i])
                    test_loss_sum += test_loss

                if self.args.fd == "full":
                    print("Epoch" + str(_) + " Example" + str(i) + ".Train loss: ",
                          str(round(train_loss, 6)))  # prints loss for each example

            # save history of train loss for later use
            mean_loss_one_epoch_train = train_loss_sum / len(self.target_train)
            self.train_loss_history.append(mean_loss_one_epoch_train)
            self.x_train_loose.append(_)

            # save history of test loss for later use
            mean_loss_one_epoch_test = test_loss_sum / len(self.target_test)
            self.test_loss_history.append(mean_loss_one_epoch_test)

            # prints train loss
            if self.args.fd == "intermediate" or self.args.fd == "full":
                # when feedback=strong activate we want a little bit more space between the messages
                if self.args.fd == "full":
                    print(" ")
                    print("Epoch" + str(_) + " Mean-train loss: " + str(
                        round(mean_loss_one_epoch_train, 6)))  # prints mean-loss of every Epoch
                    print(" ")
                else:
                    print("Epoch" + str(_) + " Mean-train loss: " +
                          str(mean_loss_one_epoch_train))  # prints mean-loss of every Epoch

            end_time = time.time()
            self.evaluation_time = end_time - start_time


    # a getter for the viszulation function
    def getter_viszualtion(self):
        return self.weights, self.train_loss_history, self.test_loss_history, self.evaluation_time, self.data_train, self.target_train, self.x_train_loose

    # saves weight and bias
    def save(self) -> None:
        filename = "polynomial_regression_housing_weights.txt"
        with open(filename, "w+", newline='') as writeFile:
            for i in self.weights:
                writeFile.write(str(i) + "\n")

        writeFile.close()

    # predicting with the model
    def predic(self, visualize_process, args_normalization) -> None:
        df_range = args_normalization[0]
        df_mean = args_normalization[1]

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
                for i in range(0,3,1):
                    # exits while loop when right inputs got inserted
                    while True:
                        input_var = input()

                        if input_var == "quit" or input_var == "Quit":
                            if visualize_process.is_alive():
                                try:
                                    visualize_process.terminate()
                                except Exception as e:
                                    print("Error: ", str(e))
                            print(" ")
                            print("Please be noted that this value is a estimate. I am not liable responsibly.")
                            print("For more information about the copyright of this programm look at my Github repository: ")
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
                rm_input = round(float(input_list[0]), 4)
                lstat_input = round(float(input_list[1]), 4)
                ptratio_input = round(float(input_list[2]), 4)

                # normalizing input
                rm_input_norm = (rm_input - df_mean[0]) / df_range[0]
                lstat_input_norm = (lstat_input - df_mean[1]) / df_range[1]
                ptratio_input_norm = (ptratio_input - df_mean[2]) / df_range[2]

                # predicting
                self.pred_target = self.hypothesis(self.weights, rm_input_norm, lstat_input_norm, ptratio_input_norm, 1)

                # denormalization of output
                denorm_pred_target = round((self.pred_target * df_range[3]) + df_mean[3], 6)

                print(" ")
                print("The model predicted that a house with the values: ")
                print("RM :" + str(rm_input))
                print("LSTAT :" + str(lstat_input))
                print("PTRATIO :" + str(ptratio_input))
                print(" ")

                # check if predicted output is negative
                if denorm_pred_target < 0:
                    print("-----------------------------------------------------------------------------")
                    print("Warning: the input values doesn't correspond to a real house.")
                    print("-----------------------------------------------------------------------------")
                    print(" ")
                else:
                    print("-----------------------------------------------------------------------------")
                    print("Is worth about: " + str(denorm_pred_target) + " in 10,000$(GER 10.000$).")
                    print("-----------------------------------------------------------------------------")
                    print(" ")

            except Exception as e:
                print("Something went wrong: ", str(e))
