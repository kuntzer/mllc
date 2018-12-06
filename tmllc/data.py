import os
import numpy as np
import batman
from astropy.table import Table
from datetime import datetime
import pylab as plt
import glob
from keras.utils import Sequence

from . import io

import logging
logger = logging.getLogger(__name__)

class PlanetPropertiesGenerator():
    
    def __init__(self, seed=None):
        
        if not seed is None:
            np.random.seed(seed)
            
    def period(self, size=1):
        
        proba = np.random.uniform(size=size)
        
        idsS = np.where(proba < 0.65)
        idsL = np.where(proba >= 0.65)
        
        #proba[idsS] = np.random.uniform(0.25, 10, size=len(idsS[0]))
        #proba[idsL] = np.random.uniform(10, 30, size=len(idsL[0]))
        proba[idsS] = np.random.uniform(20, 28.625, size=len(idsS[0]))#This is the easy version
        proba[idsL] = np.random.uniform(20, 28.625, size=len(idsL[0]))#This is the easy version
        
        return proba
    
    def radius(self, size=1):
        
        #return np.random.uniform(0.015, 0.075, size=size) #This is the easy version
        return np.random.uniform(0.005, 0.075, size=size)
        
    def aOverRstar(self, size=1):
        
        #return np.random.lognormal(np.log(25), 0.6, size=size)
        return np.random.lognormal(np.log(30), 0.6, size=size)
    
    def inc(self, size=1):
        
        return np.ones(size) * 90.
        #return np.random.uniform(88., 90., size=size)
    
    def darkLimbCoeff(self, size=1):
        
        return np.random.uniform(0., 0.6, size=size)
    
    def eccentricity(self, size=1):
        #return np.zeros(size)
        return np.random.lognormal(mean=1e-3, sigma=1.2, size=size) / 650.
        """
        print(np.amin(prop), np.mean(prop), np.amax(prop))
        plt.figure()
        
        bins = np.logspace(-6,0, 50)
        plt.hist(prop, bins=bins, log=True)
        plt.xscale('log')
        plt.show()
        """
        
    def lonPeriastron(self, size=1):
        #return 90. * np.ones(size)
        return np.random.uniform(low=0, high=180, size=size)

def generateFakeData(ntransits, nnontransit, sigmaPhoton, saveDir="data/fake/", maxDataPerFile=1024):
    """
    This generates noisy transits and another series of data, which are then saved to 3 files: 
    """

    def save2File(idFile, fluxes):
        logger.info("Preparing to save...")
        fluxes = np.array(fluxes).T
        
        io.pickleWrite(fluxes, os.path.join(saveDir, "flux_{:03d}.pkl".format(idFile)))
        logger.info("Wrote flux file nb {:03d}".format(idFile))

    logger.info("Preparing to generate {} transits and {} non-transiting LCs".format(ntransits, nnontransit))

    timeStart = datetime.now()
    
    prop = PlanetPropertiesGenerator()
    
    Nexp = 20610
    #Nexp = 1000
    paramsTruths = []
    fluxes = []

    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    # Transiting LCs
    stepShout = 5
    lastShoutOut = -stepShout
    
    logger.info("Generating {} transits LCs. This is going to take some time...".format(ntransits))
    currentid = 0
    idFile = 0
    
    for ii in np.arange(ntransits):
        
        if (ii + 1)/ntransits * 100 > lastShoutOut + stepShout:
            shoutOutval = (ii + 1)/ntransits * 100
            logger.info("Generation at {:2.0f}%".format(shoutOutval))
            lastShoutOut += stepShout
        
        params = batman.TransitParams()       #object to store transit parameters
        params.per = prop.period() #orbital period
        params.rp = prop.radius()                      #planet radius (in units of stellar radii)
        params.t0 = np.random.uniform(0, 14.)#time of inferior conjunction
        params.a = prop.aOverRstar()                        #semi-major axis (in units of stellar radii)
        params.inc = prop.inc()                      #orbital inclination (in degrees)
        params.ecc = prop.eccentricity()                       #eccentricity
        params.w = prop.lonPeriastron()                       #longitude of periastron (in degrees)
        params.limb_dark = "quadratic"        #limb darkening model
        params.u = [prop.darkLimbCoeff(), prop.darkLimbCoeff()]      #limb darkening coefficients
        
        paramsTruths.append([currentid, idFile, params.per[0], params.rp[0], params.t0, params.a[0], params.inc[0], params.ecc, params.w, params.u[0][0], params.u[1][0]])
        
        # TESS SAMPLING is 2 minutes
        # We'll take the 20610 points as given in the ETE6 (even if it translates to 28.625 days)
        #t = np.linspace(0., 5., Nexp)  #times at which to calculate light curve
        t = np.linspace(0., 28.625, Nexp)  #times at which to calculate light curve
        m = batman.TransitModel(params, t, exp_time=2)    #initializes model
        flux = m.light_curve(params)                    #calculates light curve
        fluxWNoise = np.random.normal(0., sigmaPhoton, flux.shape) + flux
        
        fluxWNoise = np.hstack([currentid, fluxWNoise])
        
        fluxes.append(fluxWNoise)
        currentid += 1 
        
        if np.shape(fluxes)[0] >= maxDataPerFile:

            save2File(idFile, fluxes)
            
            # HK ----------------------------------------
            idFile += 1
            fluxes = []
    
    if np.shape(fluxes)[0] > 0:
        save2File(idFile, fluxes)
        idFile += 1
    del fluxes 
    
    #Normals LCs
    idListNon = np.arange(currentid, currentid+nnontransit)

    startId = 0
    totgen = 0
    truthIds = np.ones(nnontransit) * -1
    
    while startId < nnontransit:
        endId = startId + maxDataPerFile
        nn = maxDataPerFile
        
        if nn + totgen > nnontransit:
            nn = nnontransit - totgen
            
        logger.info("Generating {} non-transiting LCs ({}/{} already done).".format(nn, totgen, nnontransit))
        fluxesNonTransits = 1. + np.random.normal(scale=sigmaPhoton, size=(Nexp, nn))
        id2Add = idListNon[startId:endId].reshape((len(idListNon[startId:endId]), 1)).T
        fluxesNonTransits = np.vstack([id2Add, fluxesNonTransits])
        
        truthIds[startId:endId] = idFile
        io.pickleWrite(fluxesNonTransits, os.path.join(saveDir, "flux_{:03d}.pkl".format(idFile)))
        
        totgen += nn        
        startId = endId
        idFile += 1
    
    paramsTruths = np.array(paramsTruths)    
    zeroTable = -1. * np.ones((nnontransit, paramsTruths.shape[1]))
    zeroTable[:,0] = idListNon
    zeroTable[:,1] = truthIds
    paramsTruths = np.vstack([paramsTruths, zeroTable])
    
    t = Table(paramsTruths, 
              names=('id', 'idFile', 'orbitalPeriod', 'radiusPlanetRatio', 'timeInferiorConjunction', 'a', 
                     'inclination', 'eccentricity', 'raan', 'darkLimbCoeff0', 'darkLimbCoeff1'))
    fnameTransit = os.path.join(saveDir, "truthTransits.fits")
    logger.info("Truth file saved to {}".format(fnameTransit))
    t.write(fnameTransit, format='fits', overwrite=True)
    
    logger.info("Generation done in {}.".format(datetime.now()-timeStart))
    
def generateSets(sets, trainingSetFrac=0.6, validationSetFrac=0.2):
    """
    Generate the 3 sets (training, validation and test) on a number of dataset.
    
    :param sets: a list of numpy array
    :param truthTable: the truth table used to find the truth in the truth table
    :param trainingSetFrac: fraction of the set to put in the training set
    :param validationSetFrac: fraction of the set to put in the validation set (can be set to 0!)
    """
    #TODO: Handle the fraction better than this ;)

    if not (0 < trainingSetFrac < 1 and 0 <= validationSetFrac < 1 and 0 <= trainingSetFrac + validationSetFrac <= 1):
        raise RuntimeError("Please set the fraction of the data for the different sets correctly")

    lenS = 0
    for ii, s in enumerate(sets):
        if lenS == 0:
            lenS = np.shape(s)[0]
        else:
            if not lenS == np.shape(s)[0]:
                raise RuntimeError("Set nb {} has length {}, which is not the same as {}".format(ii+1, np.shape(s)[0], lenS))
    
    ids = np.arange(lenS)
    
    idsTrain = np.random.choice(ids, size=int(lenS * trainingSetFrac), replace=False)
    ids = np.setdiff1d(ids, idsTrain)

    if validationSetFrac > 0:
        idsValid = np.random.choice(ids, size=int(lenS * validationSetFrac), replace=False)
        ids = np.setdiff1d(ids, idsValid)

    np.random.shuffle(ids)
    idsTests = ids
    ids = np.setdiff1d(ids, idsTests)
    
    assert np.size(ids) == 0
    
    train = []
    valid = []
    tests = []
    for s in sets:
        train += [s[idsTrain]]
        if validationSetFrac > 0:
            valid += [s[idsValid]]
        else:
            valid += [np.array([])]
        tests += [s[idsTests]]

    return train, valid, tests

class DataFromFile(Sequence):
    """
    Class that uses keras.Sequences (bascially a nice wrapper of a generator) to load a lot of data
    in training or validation.
    """
    
    def __init__(self, featureDir, dataset, batchSize):
        self.featureDir = featureDir
        self.dataset = dataset
        self.batchSize = batchSize
        
        labelsFiles = glob.glob(os.path.join(featureDir, "{}Labels_*.pkl".format(dataset)))
        labelsFiles.sort()
        self.labelsFiles = labelsFiles
    
        FeaturesFiles = glob.glob(os.path.join(featureDir, "{}Features_*.pkl".format(dataset)))
        FeaturesFiles.sort()
        self.FeaturesFiles = FeaturesFiles
        
        self._prepare()
        
    def _getLenData(self):
        n = 0
        for fn in self.labelsFiles:
            n += len(io.pickleRead(fn))
        if self.batchSize is None:
            self.batchSize = n
        return n
        
    def _prepare(self):
        self.fileId = 0
        self.lastLoaded = None
        self.nFeatureInFile = None
        self.featuresCounter = 0
        self.lastStop = 0 
        self.nTotSamples = self._getLenData()
        
    def __len__(self):
        return np.int(np.ceil(self.nTotSamples / self.batchSize))
    
    def loadData(self):
        self.labelsData = io.pickleRead(self.labelsFiles[self.fileId])
        self.FeaturesData = io.pickleRead(self.FeaturesFiles[self.fileId]).T
        self.lastLoaded = self.fileId
        self.nFeatureInFile = len(self.labelsData)
        assert self.nFeatureInFile == np.shape(self.FeaturesData)[0]
        
    def returnData(self, features, labels):
        if len(labels) == 1:
            features = features.reshape((1, 1, np.size(features)))
        return (features, labels)
    
    def __getitem__(self, idx):
        labels = []
        features = None
        
        #print("New call", self.featuresCounter, self.nTotSamples)
        if self.featuresCounter >= self.nTotSamples:
            #print("Got to the end of the samples for {}...".format(self.dataset))
            self._prepare()
        
        if not self.lastLoaded == self.fileId:
            self.loadData()
        
        for jj in np.arange(self.featuresCounter, self.featuresCounter + self.batchSize):
            
            self.featuresCounter += 1
            #print(jj, self.nFeatureInFile)
            
            ii = jj - self.lastStop
            if ii >= self.nFeatureInFile:
                if self.fileId + 1 < len(self.labelsFiles):
                    self.lastStop = self.nFeatureInFile
                    ii -= self.lastStop
                    self.fileId += 1
                    #print("NewfileId", self.fileId)
                    self.loadData()
                    self.lastStop = jj
                else:
                    return self.returnData(features, labels)
            
            label = self.labelsData[ii]
            feature = self.FeaturesData[ii]
            
            labels.append(label)
            feature = feature.reshape((1, 1, np.size(feature)))
            
            if features is None:
                features = feature
            else:
                features = np.concatenate([features, feature])
                
            if len(labels) >= self.batchSize:
                return self.returnData(features, labels)

