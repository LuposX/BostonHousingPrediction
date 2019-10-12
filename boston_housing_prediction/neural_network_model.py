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

        # misc
        self.weight = self.init_weight()
        self.layers_history = []
        self.output_neuron_activ = 0

    # init the weights of the nn
    def init_weight(self) -> list:
        weights = []

        # for input layer
        weights.append(np.ones(self.num_feature * self.list_architecture[0]).tolist())

        # for hidden layers
        for i in range(0, len(self.list_architecture)):
            if len(self.list_architecture) > i + 1:
                weights.append(np.ones(self.list_architecture[i] * self.list_architecture[i + 1]).tolist())

        # for output layer
        weights.append(np.ones(1 * self.list_architecture[-1]).tolist())

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
        # --------------------------------------------
        # input layer is outside of loop
        input_layer_trans = np.reshape(input_layer, (1, self.num_feature)).flatten()
        weights = np.asarray(np.reshape(self.weight[0], (self.num_feature, self.list_architecture[0])))

        # calculating
        curr_layer = np.dot(input_layer_trans, weights)  # curr_layer current layer

        # adding to history for later use
        self.layers_history.append(curr_layer)

        for k in range(len(self.list_architecture) - 1):  # "- 1" because we subtract output layer
            # getting input values in right shape
            weights = np.asarray(np.reshape(self.weight[k + 1], (self.list_architecture[k], self.list_architecture[k + 1])))  # "weight[k + 1]" because weight[0] is for input layer

            # calculating
            curr_layer = np.dot(curr_layer, weights)
            activ_curr_layer = relu(curr_layer)

            # adding to history for later use
            self.layers_history.append(activ_curr_layer)

            # communication is key
            print(" ")
            print("------------")
            print("weights:", weights)
            print("shape:", weights.shape)

            print(" ")
            print("input:", curr_layer)
            print("shape:", curr_layer.shape)

            print(" ")
            print("output:", activ_curr_layer)

        # output layer is outside of loop
        weights = np.asarray(np.reshape(self.weight[-1], (1, self.list_architecture[-1])))

        # calculating
        output_neuron = np.dot(self.layers_history[-1], weights)

        self.output_neuron_activ = linear(output_neuron)  # activate layer

        # adding to history for later use
        self.layers_history.append(self.output_neuron_activ)

        print("Final output:", self.output_neuron_activ)

    def backward(self):
        pass
