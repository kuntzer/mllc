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
    logger.info("Wrote %s" % filepath)

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
    
    logger.info("Read %s" % filepath)
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

def saveSets(saveDir, train, valid, tests, overwrite=False):
    
    # Looking at the directory existence. Behaviour according to user's choice
    if os.path.exists(saveDir):
        if overwrite: 
            shutil.rmtree(saveDir)
        else:
            raise IOError("Directory {} exists already. Choose a different run name".format(saveDir))
    saveDir = os.path.join(saveDir, "data")
    os.makedirs(saveDir)
    
    # TODO: replace this. I don't like have to hard-code the different datasets here.
    sets = {"train": train, "valid":valid, "tests":tests}
    
    for key in sets.keys():
        logging.info("Saving {} features and labels to {}".format(key, saveDir))
        array, label, truthTable = sets[key]
        logging.info("with {} features and {} samples".format(np.shape(array)[1], np.shape(array)[0]))
        
        array = array.reshape(array.shape[0], 1, array.shape[1])
        pickleWrite(array, os.path.join(saveDir, "{}Features.pkl".format(key)))
        pickleWrite(label, os.path.join(saveDir, "{}Labels.pkl".format(key)))
        truthTable.write(os.path.join(saveDir, "{}TruthTable.fits".format(key)), format='fits') 
        
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

