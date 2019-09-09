import os
from os import path
import urllib.request


def is_non_zero_file():
    return (os.path.isfile("housing.csv") and os.path.getsize("housing.csv") > 0) or (
            os.path.isfile("boston_housing.csv") and os.path.getsize("boston_housing.csv") > 0)


def downloading():
    try:
        file = open("boston_housing.csv", "w+")  # the plus asgins when not exist we create it
        data = urllib.request.urlopen(
            "https://raw.githubusercontent.com/udacity/machine-learning/master/projects/boston_housing/housing.csv").read().decode(
            'utf-8')
        file.write(data)
        file.close()
    except:
        try:
            file = open("boston_housing.csv", "w+")  # the plus asgins when not exist we create it
            data = urllib.request.urlopen(
                "https://raw.githubusercontent.com/LuposX/BostonHousingPrediction/master/dataset/boston_housing.csv").read().decode(
                'utf-8')
            file.write(data)
            file.close()
        except Exception as e:
            print("Error: ", str(e))


# downloding the dataset if not avaible
if is_non_zero_file():
    print("exist")
else:
    downloading()
