from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import csv

# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=33))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.load_weights('sampleWeightsNew.h5')
# Generate dummy data
data = np.empty([509444, 33])
testData = np.empty([509444, 33])
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

# Train the model, iterating on the data in batches of 32 samples


history = model.fit(data, labels, validation_split=0.25, epochs=200, batch_size=128)



model.save_weights('sampleWeightsLargeTest.h5')
model.save('testModel.h5')



print(model.evaluate(x=testData, y=labels, batch_size=1))