import numpy as np
import os
import logging
from datetime import datetime
#from sklearn.preprocessing import MinMaxScaler

import tmllc
import matplotlib.pyplot as plt

###################################################################################################
def preprocessNonParametric(data):
    logging.critical("NORMALIZATION IS CRITICAL!")
    data -= 1.
    maxPerSignal = np.amax(data, axis=0)
    data /= maxPerSignal
    return data

logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.INFO)

fakeDataDir = "data/fakeWideParams"
runName = "fakeWideParams"
saveDir = os.path.join("runs", runName)
maxDataPerFile = 200
# This, is set to True, will reorder the dataset per file number. It will require less I/O operation
# but assume there is no order in the data.
allowOrderPerFile = False 
show = False

# Beware the results are not reproducible since we select always a different subset, so let's fix the numpy seed
np.random.seed(10)

###################################################################################################

timeStart = datetime.now()

logging.info("Loading fake data...")

truthTable = tmllc.io.loadTable("truthTransits", "{}/".format(fakeDataDir))
if show:
    idKeep = truthTable["orbitalPeriod"] > 0.
    truthTableShow = truthTable[idKeep]
    for colname in truthTable.colnames:
        plt.figure()
        plt.title(colname)
        plt.hist(truthTableShow[colname], 25)
    plt.show()
    del truthTableShow

# First decide on the order, then we will load the files...

sets = [truthTable]
train, valid, tests = tmllc.data.generateSets(sets, trainingSetFrac=0.8, validationSetFrac=0.2)
names = ["train", "valid", "tests"]
saveDir = tmllc.io.prepare2SaveSets(saveDir=saveDir, overwrite=True)

for arr, key in zip([train, valid, tests], names):
    arr = arr[0]
    tmllc.io.saveSet(saveDir, dataDir=fakeDataDir, arr=arr, key=key, truthTable=truthTable, preProcessFun=preprocessNonParametric, allowOrderPerFile=allowOrderPerFile, maxDataPerFile=maxDataPerFile)

logging.info("Pre-processed training data in {}".format(datetime.now()-timeStart))
