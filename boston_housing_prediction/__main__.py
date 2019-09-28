'''
Author: Lupos
Started: 08.09.2019
Lang: Phyton
Description: Prediction of boston housing market prices.
version: 0.2.2

Dataset:
Housing Values in Suburbs of Boston

RM: average number of rooms per dwelling(Wohnung)
LSTAT: percentage of population considered lower status
PTRATIO: pupil-teacher ratio by town
MEDV: median value of owner-occupied homes in 10.000$

Latest change:
- fixed Normalization for data
- organisational stuff and bugfixes: #24
- negative input/output
'''

# My Files that get imported
import boston_main

# TODO: fix train and test loss

if __name__ == "__main__":
    boston_main.main()
