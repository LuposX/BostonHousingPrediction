import numpy as np
from misc_libary import sigmoid, relu, linear

# TODO: change wwight insit from ones() to zero()

class NeuralNetwork:
    def __init__(self, list_architecture, df):
        # Hyperparameter
        self.list_architecture = list_architecture
        self.epochs = 20
        self.alpha = 0.003
        self.num_feature = 3

        # get the data in form
        self.df_data_train = df[0].iloc[:,  df[0].columns != "MEDV"].reset_index(drop=True)    # the ":" stands for every element in there
        self.df_target_train = df[0]["MEDV"].tolist()

        self.df_data_test = df[1].iloc[:, df[0].columns != "MEDV"].reset_index(drop=True)  # the ":" stands for every element in there
        self.df_target_test = df[1]["MEDV"].tolist()

        # weights
        self.weight = self.init_weight()

    # init the weights of the nn
    def init_weight(self) -> list:
        weights = []

        # for input layer
        weights.append(np.ones(self.num_feature * self.list_architecture[0]).tolist())

        for i in range(0, len(self.list_architecture)):
            if len(self.list_architecture) > i + 1:
                weights.append(np.ones(self.list_architecture[i] * self.list_architecture[i + 1]).tolist())

        return weights



    def train(self):
        for i in range(self.epochs):
            pass

    # calulating the forward pass
    def forward(self):
        # for _ in range(self.epochs):
        #for i in range(len(self.df_data_train)):
        # input layer
        #input_layer = [self.df_data_train["RM"][i], self.df_data_train["LSTAT"][i], self.df_data_train["PTRATIO"][i]]
        input_layer = [self.df_data_train["RM"][2], self.df_data_train["LSTAT"][2], self.df_data_train["PTRATIO"][2]]
        #activ_input_layer = relu(input_layer)  # "activ" stands for activated

        # for amount of hidden layer
        #for k in range(len(list_architecture) - 2):  # "- 2" because we subtract input and output layer
        hidden_layer = np.dot(input_layer, np.reshape(self.weight[0], (self.num_feature, self.list_architecture[0])))

        # communication is key
        print("weights: ")
        print(self.weight[0])
        print("input: ")
        print(input_layer)

        print(" ")
        print("output")
        print(hidden_layer)

        # output layer
        #output_neuron = linear()

        print(hidden_layer)


    def backward(self):
        pass
