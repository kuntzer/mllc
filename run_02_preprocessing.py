import numpy as np
import os
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import tmllc

logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.INFO)

useFakeData = True
fakeDataDir = "data/fakeLarge"
runName = "fakeDataLarge"
saveDir = os.path.join("runs", runName)

# Beware the results are not reproducible since we select always a different subset, so let's fix the numpy seed
np.random.seed(10)

timeStart = datetime.now()

if useFakeData:
    logging.info("Loading fake data...")
    fluxNone = tmllc.io.pickleRead("{}/fluxNonTransits.pkl".format(fakeDataDir))
    fluxPlanet = tmllc.io.pickleRead("{}/fluxTransits.pkl".format(fakeDataDir))
    truthTable = tmllc.io.loadTable("truthTransits", "{}/".format(fakeDataDir))
else:
    logging.info("Loading 'real' data...")
    tics, timestamps, fluxPlanet, _ = tmllc.io.loadDataset("planet")
    _, _, fluxNone, _ = tmllc.io.loadDataset("none")
    
    # Select the same number of planets and nothing LC to avoid biaising the network
    ids = np.random.choice(np.arange(fluxNone.shape[1]), fluxPlanet.shape[1])
    fluxNone = fluxNone[:,ids]
    
flux = np.hstack([fluxPlanet, fluxNone])
labels = np.hstack([np.ones(np.shape(fluxPlanet)[1]), np.zeros(np.shape(fluxNone)[1])])
# Let's free up some memory here
del fluxPlanet, fluxNone

# Normalise the curves
print(np.amax(flux))
print(np.amin(flux))

# TODO: save the scaler also
# TODO: may be that's not the right way to normalise. 
# TODO: you should look at whether you should not normalise by sigma noise.
scaler = MinMaxScaler()
#TODO:
#flux = scaler.fit_transform(flux)
logging.critical("NORMALIZATION IS CRITICAL!")

# and preprocess now
flux -= 1.
maxPerSignal = np.amax(flux, axis=0)
print(maxPerSignal.shape)
print(flux / maxPerSignal)
flux = flux / maxPerSignal
print(np.amax(flux, axis=0))
#exit()
#flux *= -1e2
#flux /= np.nanmedian(flux, axis=0)
print(np.amax(flux))
print(np.amin(flux))

sets = flux.T, labels, truthTable

train, valid, tests = tmllc.data.generateSets(sets, trainingSetFrac=0.8, validationSetFrac=0.2)

tmllc.io.saveSets(saveDir, train, valid, tests, overwrite=True)

logging.info("Pre-processed training data in {}".format(datetime.now()-timeStart))
