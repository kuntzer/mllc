from datetime import datetime
import os
import logging
import numpy as np
import glob

import tmllc
import models

logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.INFO)

runName = "fakeWideParams"


saveDir = os.path.join("runs", runName)
np.random.seed(10)

timeStart = datetime.now()

# Let's get the size of the feature by randomly loading one.
FeaturesFiles = glob.glob(os.path.join(saveDir, "data", "trainFeatures_*.pkl"))
dimFeature = tmllc.io.pickleRead(FeaturesFiles[0])
dimFeature = np.shape(dimFeature)[0]

# Let's define the model, the input shape is directly derived from the trainFeatures
modelFct = models.RNNOnly
modelName = modelFct.__name__
logging.info("Model name is selected to be: {}".format(modelName))
modelDir = os.path.join("runs", runName, modelName)
if os.path.exists(modelDir):
    raise IOError("Model name {} exists ({})".format(modelName, modelDir))
else:
    os.makedirs(modelDir)
model = modelFct(dimFeature)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("*" * 50)
fitHistory = model.fit_generator(generator=tmllc.data.DataFromFile(os.path.join(saveDir, "data"), dataset="train", batchSize=64),
            epochs=4,
            validation_data=tmllc.data.DataFromFile(os.path.join(saveDir, "data"), dataset="valid", batchSize=None)
            )


logging.info("Training the model took {}".format(datetime.now()-timeStart))

tmllc.io.saveModel(model, fitHistory, modelDir)

