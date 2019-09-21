import time
from misc_libary import loss
import sys

class PolynomialRegression:
    def __init__(self, df, args):
        # weight/bias init
        self.weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.bias = 1

        # how man epoch we train
        self.epochs = 30
        self.alpha = 0.000000008
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

        self.w1 = 0    # self w1 is a dummy value. can be removed later

    # our hypothesis/ what our model predicts
    def hypothesis(self, weights, f1, f2, f3, bias):
        #print(weights[0])
        # pred = round(weights[0] * f1 + weights[1] * f1 ** 2 + weights[2] * f2 + weights[3] * f2 ** 2 + \
        #              weights[4] * f3 + weights[5] * f3 ** 2 + weights[6] * bias, 10)
        pred = round(weights[0] * f1 + weights[1] * f1 ** 2 + weights[2] * f1 ** 3 + weights[3] * f2 + weights[4] * f2 ** 2 + \
                          weights[5] * f2 ** 3 + weights[6] * f3 + weights[7] * f3 ** 2 + weights[8] * f3 ** 3 + weights[9] * bias, 10)
        return pred

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

            for i in range(len(self.target_train)):
                # our features and their data
                f1 = self.data_train["RM"][i]
                f2 = self.data_train["LSTAT"][i]
                f3 = self.data_train["PTRATIO"][i]

                # our hypothesis/ what our model predicts
                pred_target = self.hypothesis(self.weights, f1, f2, f3, self.bias)

                # update our weights
                #print((pred_target - self.target_train[i]))

                # try:
                #     error = (pred_target - self.target_train[i])
                #     if error > 1000000000000:
                #         error = 1000000000000
                #     elif error < -1000000000000:
                #         error = -1000000000000
                # except RuntimeWarning or RuntimeError():
                #     if error > 1000000000000:
                #         error = 1000000000000
                #     elif error < -1000000000000:
                #         error = -1000000000000
                error = (pred_target - self.target_train[i])

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
                #print((pred_target - self.target_train[i]) * self.bias)
                # sums train loss
                train_loss = loss(pred_target, self.target_train[i])
                train_loss_sum += train_loss

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

                    # evalutae with loss
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
                        round(mean_loss_one_epoch_test, 6)))  # prints mean-loss of every Epoch
                    print(" ")
                else:
                    print("Epoch" + str(_) + " Mean-train loss: " +
                          str(mean_loss_one_epoch_test))  # prints mean-loss of every Epoch

            end_time = time.time()
            self.evaluation_time = end_time - start_time


    # a getter for the viszulation function
    def getter_viszualtion(self) -> list:
        return self.weights, self.train_loss_history, self.test_loss_history, self.evaluation_time, self.data_train, self.target_train, self.x_train_loose

    # saves weight and bias
    def save(self) -> None:
        filename = "polynomial_regression_housing_weights.txt"
        with open(filename, "w+", newline='') as writeFile:
            for i in self.weights:
                writeFile.write(str(i) + "\n")

        writeFile.close()

    # predicting with the model
    def predic(self, visualize_process) -> None:
        time.sleep(
            1)  # sleeps so that the function visualize()(which is seperate process through multiprocessing) has enough time to print the output correctly
        self.pred_target = 0
        print(" ")
        print("Prediction")
        print("------------------------------------")
        print("With this model you can predict how much a house is worth.")
        # while true until valid input
        while True:
            try:
                # get input for our model
                print("If you want to quit type: 'quit'.")
                print("Please enter the RM vaule. Values with the type of Int or float are only allowed.")
                rm_input = input()

                if rm_input == "quit" or rm_input == "Quit":
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

                print(" ")

                print("Please enter the LSTAT vaule. Values with the type of Int or float are only allowed.")
                lstat_input = input()
                print(" ")

                print("Please enter the PTRATIO vaule. Values with the type of Int or float are only allowed.")
                ptratio_input = input()
                print(" ")

                rm_input = round(float(rm_input), 4)
                lstat_input = round(float(lstat_input), 4)
                ptratio_input = round(float(ptratio_input), 4)

                self.pred_target = self.hypothesis(self.weights, rm_input, lstat_input, ptratio_input, 1)

                print(" ")
                print("The model predicted that a house with the values: ")
                print("RM :" + str(rm_input))
                print("LSTAT :" + str(lstat_input))
                print("PTRATIO :" + str(ptratio_input))
                print(" ")
                print("Is worth about: " + str(round(self.pred_target, 4)) + " in 10,000$(GER 10.000$).")
                print(" ")
            except ValueError:
                print("Invalid Input!")
