import os
import pandas as pd
from os import path


def get_Data() -> object:
    # specify in which directories the dataset could be in
    directories_to_try = ["/datasets", "C:/Users/" + str(os.getlogin()) + "/Downloads/", "../datasets", "/dataset",
                          "../dataset", "../", ""]

    # searching for the file
    for candidate_directory in directories_to_try:
        try:
            if path.exists(str(candidate_directory) + "boston_housing.csv"):
                print(str(candidate_directory) + "boston_housing.csv")
                df = pd.read_csv(str(candidate_directory) + "boston_housing.csv")
                break
        except:
            try:
                if path.exists(str(candidate_directory) + "housing.csv"):
                    print(str(candidate_directory) + "housing.csv")
                    df = pd.read_csv(str(candidate_directory) + "housing.csv")
                    break
            except FileNotFoundError:
                 print("oops, directory doesn't exist")
    else:  # this executes if the loop completes without ever encountering a `break`
        print("Could not find acceptable directory!")
    return df

df = get_Data()
print(df.head())

