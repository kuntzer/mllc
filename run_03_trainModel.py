from datetime import datetime
import os
import logging
import numpy as np

import tmllc

import models

logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.INFO)

runName = "fakeWideParams"


saveDir = os.path.join("runs", runName)
np.random.seed(10)

timeStart = datetime.now()

trainFeatures = tmllc.io.loadDataset("trainFeatures", os.path.join(saveDir, "data"))
trainLabels = tmllc.io.loadDataset("trainLabels", os.path.join(saveDir, "data"))
validFeatures = tmllc.io.loadDataset("validFeatures", os.path.join(saveDir, "data"))
validLabels = tmllc.io.loadDataset("validLabels", os.path.join(saveDir, "data"))

#Tensorflow uses [samples][height][width][channels] order, while Theano is in reverse order.

#trainFeatures = np.reshape(trainFeatures, (3200, 1000)).T
print(trainFeatures.shape)


# Let's define the model, the input shape is directly derived from the trainFeatures
modelFct = models.RNNOnly
modelName = modelFct.__name__
logging.info("Model name is selected to be: {}".format(modelName))
modelDir = os.path.join("runs", runName, modelName)
if os.path.exists(modelDir):
    raise IOError("Model name {} exists ({})".format(modelName, modelDir))
else:
    os.makedirs(modelDir)
model = modelFct(trainFeatures)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

batch_size = 128
fitHistory = model.fit(trainFeatures, trainLabels, validation_data=(validFeatures, validLabels), epochs=7, batch_size=batch_size)

logging.info("Training the model took {}".format(datetime.now()-timeStart))

tmllc.io.saveModel(model, fitHistory, modelDir)

