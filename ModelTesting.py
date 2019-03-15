from keras.models import load_model
import numpy as np
import csv
import matplotlib.pyplot as plt

model = load_model('testModel.h5')

testData = np.empty([509444, 33])
data = np.empty([509444, 33])
with open('WRegularSeasonDetailedResults.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    i = 0
    k = 0;
    j = 0;
    for row in csvReader:
        if i == 0:
            print(row)
        else:
            if i%2 == 0:
                data[k] = row
                k += 1
            else:
                if i%2 == 1:
                    testData[j] = row
                    j += 1
        i += 1

labels = np.empty([509444, 2])
i = 0
while i < len(labels):
    labels[i] = [0, 1]  # Low number for first = first team id wins
    i += 1


print(model.evaluate(x=testData, y=labels, batch_size=1))

