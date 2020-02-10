import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms

import pandas as pd

import tqdm

from datetime import datetime
import sys


class BostonDataset(Dataset):
    """Boston Housing dataset"""

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        data = df.loc[:, df.columns != "MEDV"]
        self.data = torch.tensor(data.values, dtype=torch.float32)

        target = df.loc[:, df.columns == "MEDV"]
        target /= 1000000
        self.target = torch.tensor(target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample_target = self.target[idx]

        return sample, sample_target



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=6)
        self.fc2 = nn.Linear(in_features=6, out_features=3)
        self.fc3 = nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        x = x
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def get_input_parameters():
    # getting the learning rate for the model
    alpha = 0
    epochs = 0
    batch_size = 0
    while True:
        try:
            # getting the learning rate
            alpha = input("Please type the value of learning rate you want to use: ")

            alpha = float(alpha)
            if 0 < alpha < 1:
                alpha = alpha
                break
            print(" ")
            print("Please input a number between 0 and 1 :)")
        except ValueError:
            print(" ")
            print("Invalid Input!")

    # exits while loop when right inputs got inserted
    while True:
        try:
            # get input for our model
            epochs = input("Please type the numbers of epoch you want to train: ")

            epochs = int(epochs)
            if epochs > 0:
                epochs = epochs
                break
            print(" ")
            print("Please don't input negative numbers :)")
        except ValueError:
            print(" ")
            print("Invalid Input!")

        # exits while loop when right inputs got inserted
    while True:
        try:
            # get input for our model
            batch_size = input("Please type the batch_size: ")

            batch_size = int(batch_size)
            if batch_size > 0:
                batch_size = batch_size
                break
            print(" ")
            print("Please don't input negative numbers :)")
        except ValueError:
            print(" ")
            print("Invalid Input!")

    return alpha, epochs, batch_size


def _init_data(batch_size):
    train_set = BostonDataset("boston_housing.csv")
    data_loader = DataLoader(train_set, batch_size=batch_size)

    return data_loader


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def train_nn(lr, EPOCH, batch_size, save):
    print(" ")
    print("Training")
    print("---------------------------------")

    train_loader = _init_data(batch_size)

    net = NeuralNetwork()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range(EPOCH):
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            
            optimizer.zero_grad()
            pred = net(inputs)
            loss = loss_func(pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Loss in Epoch {epoch}: {total_loss}')
    print("---------------------------------")

    if save:
        time = datetime.now()
        date_time = time.strftime("_%d_%m_%Yx%H_%M_%S")
        torch.save(net.state_dict(), "../pre_trained_models/" + str(net.__class__.__name__) + str(date_time) + ".pt")

def get_input_predict():
    while True:
        try:
            print('If you want to quit type: "quit".')
            print('Only Values with the type of "int" or "float" are allowed.')
            print("Type the Values in the following order: ")
            print("1.RM 2.LSTAT 3.PTRATIO")
            input_list = []
            default_values = [6.24, 12.94,
                              18.52]  # those are the default values when field is left empty. default values corrospond to mean values of feature
            for i in range(0, 3, 1):
                # exits while loop when right inputs got inserted
                while True:
                    input_var = input() or default_values[i]

                    if input_var == "quit" or input_var == "Quit":
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

        return input_list


def predict_nn(model_path):
    net = NeuralNetwork()
    try:
        net.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(e)
        sys.exit(0)

    inputs = get_input_predict()

    out = net(torch.tensor(inputs))

    print("Output")
    print("----------------")
    print("Predicted-Output: ", out)
    print("Output in 1,000,000$")

