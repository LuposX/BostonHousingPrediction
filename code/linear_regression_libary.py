import time
from misc_libary import loss
import sys
import csv

class LinearRegression:
    def __init__(self, df, args):
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
        self.args = args

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

                if self.args.fd == "full":
                    print("Epoch" + str(_) + " Example" + str(i) + ".Train loss: ", str(round(train_loss, 6)))  # prints loss for each example

            # save history of train loss for later use
            mean_loss_one_epoch_train = train_loss_sum / len(self.data_train)
            self.train_loss_history.append(mean_loss_one_epoch_train)
            self.x_train_loose.append(_)

            # save history of test loss for later use
            mean_loss_one_epoch_test = test_loss_sum / len(self.data_test)
            self.test_loss_history.append(mean_loss_one_epoch_test)

            # prints train loss
            if self.args.fd == "intermediate" or self.args.fd == "full":
                # when feedback=strong activate we want a little bit more space between the messages
                if self.args.fd == "full":
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
    def predic(self, visualize_process) -> None:
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
                    print(" ")
                    print("Please be noted that this value is a estimate. I am not liable responsibly.")
                    print("For more information about the copyright of this programm look at my Github repository: ")
                    print("github.com/LuposX/BostonHousingPrediction")
                    sys.exit(0)  # exit the script sucessful
                    break
                else:
                    rm_input = round(float(rm_input), 4)
                    self.pred_target = self.w1 * rm_input + self.bias  # predicting

                    print(" ")
                    print("The model predicted that a house with a RM value of: " + str(rm_input) + ".")
                    print("Is worth about: " + str(round(self.pred_target, 4)) + " in 10,000$(GER 10.000$).")
                    print(" ")
            except ValueError:
                print("Invalid Input!")

    # a getter for the viszulation function
    def getter_viszualtion(self) -> list:
        weights_bias = [self.w1, self.bias]
        return weights_bias, self.train_loss_history, self.test_loss_history, self.evaluation_time, self.data_train, self.target_train, self.x_train_loose
