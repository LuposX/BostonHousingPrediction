import csv

row = [["weight_bias"], [2], [2]]
filename = "linear_regression_housing_weights.csv"
with open(filename, "w+", newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(row)

writeFile.close()