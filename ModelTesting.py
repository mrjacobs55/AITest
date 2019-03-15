from keras.models import load_model
import numpy as np
import csv

model = load_model('testModel.h5')


testData = np.empty([23168, 33])
data = np.empty([23168, 33])
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

labels = np.empty([23168, 2])
i = 0
while i < len(labels):
    labels[i] = [0, 1]  # Low number for first = first team id wins
    i += 1


print(model.evaluate(x=testData, y=labels, batch_size=64))

print('WNCAA Tournament test')
testData = np.empty([567, 33])
with open('WNCAATourneyDetailedResults.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    i = 0
    for row in csvReader:
        if i == 0:
            print(row)
        else:
            testData[i-1] = row
        i += 1

labels = np.empty([567, 2])
i = 0
while i < len(labels):
    labels[i] = [0, 1]  # Low number for first = first team id wins
    i += 1


print(model.evaluate(x=testData, y=labels, batch_size=64))

print("Mens Season Test")
testData = np.empty([41021, 33])
data = np.empty([41021, 33])
with open('MensRegularSeasonDetailedResults.csv') as csvDataFile:
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

labels = np.empty([41021, 2])
i = 0
while i < len(labels):
    labels[i] = [0, 1]  # Low number for first = first team id wins
    i += 1


print(model.evaluate(x=testData, y=labels, batch_size=64))

