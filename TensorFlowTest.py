from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import csv

newModel = False
graph = False
numSaveLoops = 2
epochs = 250
batchSize = 64
saveModel = 'newModel.h5'

# For a single-input model with 2 classes (binary classification):
if newModel:
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=33))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
else:
    model = load_model(saveModel)
#    model.load_weights('sampleWeightsLargeTest.h5')

# Generate dummy data
data = np.empty([23168, 33])
testData = np.empty([23168, 33])
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

# Train the model, iterating on the data in batches of 32 samples
gen = 0
while gen < numSaveLoops:
    history = model.fit(data, labels,  epochs=epochs, batch_size=batchSize)
    model.save_weights('sampleWeightsLargeTest.h5')
    model.save(saveModel)
    gen += 1

    plt.plot(history.history['acc'])

if graph:
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
