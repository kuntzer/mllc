import os
import shutil
import numpy as np
import pickle
import gzip
from keras.models import model_from_json
import astropy.table as astab 
import json

import logging
logger = logging.getLogger(__name__)

def pickleWrite(obj, filepath, protocol=-1):
    """
    I write your python object obj into a pickle file at filepath.
    If filepath ends with .gz, I'll use gzip to compress the pickle.

    :param obj: python container you want to compress
    :param filepath: string, path where the pickle will be written
    :param protocol: Leave protocol = -1 : I'll use the latest binary protocol of pickle.

    """
    if os.path.splitext(filepath)[1] == ".gz":
        pkl_file = gzip.open(filepath, 'wb')
    else:
        pkl_file = open(filepath, 'wb')

    pickle.dump(obj, pkl_file, protocol)
    pkl_file.close()
    logger.debug("Wrote %s" % filepath)

def pickleRead(filepath):
    """
    I read a pickle file and return whatever object it contains.
    If the filepath ends with .gz, I'll unzip the pickle file.

    :param filepath: string, path of the pickle to load
    :return: object contained in the pickle
    """
    if os.path.splitext(filepath)[1] == ".gz":
        pkl_file = gzip.open(filepath,'rb')
    else:
        pkl_file = open(filepath, 'rb')
    obj = pickle.load(pkl_file)
    pkl_file.close()
    
    logger.debug("Read %s" % filepath)
    return obj

def loadDataset(category, saveDir="data/"):
    """
    A straight forward helper around the pickle function
    """
    return pickleRead(os.path.join(saveDir, "{}.pkl".format(category)))

def loadTable(category, saveDir="data/"):
    """
    A straight forward helper around the pickle function
    """
    return astab.Table.read(os.path.join(saveDir, "{}.fits".format(category)))

def prepare2SaveSets(saveDir, overwrite=False):
    """
    Looking at the directory existence. Behaviour according to user's choice
    """
    
    if os.path.exists(saveDir):
        if overwrite: 
            shutil.rmtree(saveDir)
        else:
            raise IOError("Directory {} exists already. Choose a different run name".format(saveDir))
    dataDir = os.path.join(saveDir, "data")
    os.makedirs(dataDir)
    return dataDir

def saveSet(saveDir, dataDir, arr, key, truthTable, preProcessFun=None, allowOrderPerFile=False, maxDataPerFile=1024, **kwargs):
    """
    Preprocesses and saves a set. Saves a certain numnber of samples per file. Note that the preprocessing is applied per file.
    There is no possibility with this function to compute cross-files or cross-set propreties for normalisation  purposes.
    
    :param saveDir: where to save the features, labels and truthtables
    :param dataDir: where are the data files (flux files)
    :param key: what is the name of the set (e.g. train, valid or test)
    :param truthTable: the truth table from the data generation
    :param preProcessFun: function to be applied for the preprocessing
    :param allowOrderPerFile: if set to True will reorder the dataset per file number. It will require less I/O operation but assume there is no order in the data.
    :param maxDataPerFile: number of samples per files
    
    All remains kwargs are passed to `preProcessFun`
    """
    
    def save2file(saveDir, features, labels, fileId, key, preProcessFun, **kwargs):
        features = np.array(features).T
        if not preProcessFun is None:
            features = preProcessFun(features, **kwargs)
        pickleWrite(features, os.path.join(saveDir, "{}Features_{:03d}.pkl".format(key, fileId)))
        pickleWrite(labels, os.path.join(saveDir, "{}Labels_{:03d}.pkl".format(key, fileId)))
        logger.info("Wrote feature file nb {:03d}".format(fileId))
    
    fileId = 0

    if allowOrderPerFile:
        idSort = np.argsort(arr['idFile'])
        arr = arr[idSort]
    
    loadedFileNb = None
    labels = []
    features = []
    idsPreped = []
    ids = np.array(arr["id"], dtype=np.int)
    truthTable_ = truthTable[ids]
    
    stepShout = 1.
    lastShoutOut = -stepShout
    nSample = 0
    
    nsamples = len(arr)
    for lc in arr:
        if (nSample + 1)/nsamples * 100 > lastShoutOut + stepShout:
            shoutOutval = (nSample + 1)/nsamples * 100
            logger.info("Preprocessing at {:2.0f}% for {}".format(shoutOutval, key))
            lastShoutOut += stepShout
        
        if not loadedFileNb == lc["idFile"]:
            loadedFileNb = int(lc["idFile"])
        
            loadedFluxes = pickleRead("{}/flux_{:03d}.pkl".format(dataDir, loadedFileNb)).T
            loadedFluxIds = loadedFluxes[:, 0]
            loadedFluxes = loadedFluxes[:, 1:]

        if lc["orbitalPeriod"] > 0.:
            labels.append(1)
        else:
            labels.append(0)
    
        currentId = int(lc["id"])
        foundId = np.where(loadedFluxIds == currentId)[0]
        assert len(foundId) < 2
        if len(foundId) == 0:
            raise IndexError("Found no corresponding flux")
        assert len(foundId) == 1
        foundId = foundId[0]
        
        features.append(list(loadedFluxes[foundId]))
        idsPreped.append(fileId)
        
        #print(np.shape(features))
    
        if len(labels) >= maxDataPerFile:
            save2file(saveDir, features, labels, fileId, key, preProcessFun, **kwargs)
            fileId += 1

            labels = []
            features = []
            
        nSample += 1
            
    if len(labels) > 0:
        save2file(saveDir, features, labels, fileId, key, preProcessFun, **kwargs)
    truthTable_["idFileFeatures"] = idsPreped
    
    truthTable_.write(os.path.join(saveDir, "{}TruthTable.fits".format(key)), format='fits') 
    fileId += 1
        
def saveModel(model, fitHistory, saveDir, name="model"):
    """
    saves a Keras (trained) model to disk
    
    :param model: the model to save
    :param fitHistory: the output of the fit
    :param saveDir: where to save the model
    :param name: name of the disk (Default: model)
    """

    fnameJson = os.path.join(saveDir, "{}Definition.json".format(name))
    fnameh5 = os.path.join(saveDir, "{}Params.h5".format(name))
    fnameHist = os.path.join(saveDir, "{}History.pkl".format(name))

    # serialize model to JSON
    model_json = model.to_json()    
    with open(fnameJson, "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights(fnameh5)
    logging.info("Saved model to disk in directory {} with name {}".format(saveDir, name))
    
    # Save History of training
    pickleWrite(fitHistory.history, fnameHist)
    #with open(fnameHist, 'wb') as file_pi:
    #    pickle.dump(fitHistory.history, file_pi)
    
def loadModel(saveDir, name="model"):

    fnameJson = os.path.join(saveDir, "{}Definition.json".format(name))
    fnameh5 = os.path.join(saveDir, "{}Params.h5".format(name))

    # Model reconstruction from JSON file
    with open(fnameJson, 'r') as f:
        model = model_from_json(f.read())
    
    # Load weights into the new model
    model.load_weights(fnameh5)
    
    return model

def loadHistory(saveDir, name="model"):
    
    fnameHist = os.path.join(saveDir, "{}History.pkl".format(name))
    return pickleRead(fnameHist)

def jsonWrite(data, filepath, indent=4):
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    logger.info("Wrote %s" % filepath)
    
def jsonRead(filepath):
    
    with open(filepath, 'r') as f:
        datastore = json.load(f)
    logger.info("Read %s" % filepath)
        
    return datastore

