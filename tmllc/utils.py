import astropy.table as astab 
import os
from astropy.units import cds
from astropy import units as u
import glob
import numpy as np
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from . import io

def getTruthList(category, truthDir="data/truthFiles"):
    """
    Returns a list of the LC listed in the truth files of the category.
    
    :param category: eb, backeb, planet or star
    """
    
    fname = "ete6_{}_data.txt".format(category)
    fname = os.path.join(truthDir, fname)
    
    if not os.path.exists(fname):
        raise RuntimeError("Category {} is unknown".format(category))
    
    if category == 'eb':
        colnames = ["tic", "ebNumber", "orbitalPeriod", "epoch", "b", "omega", \
                    "eccentricity", "1stStarDepth", "2ndStarDepth", "contactBinAmplitude", \
                    "1stRadiusStar", "2ndRadiusStar", "tempEff", "logg"]
        units = [u.dimensionless_unscaled, u.dimensionless_unscaled, u.day, u.day, \
                 u.dimensionless_unscaled, u.dimensionless_unscaled, u.dimensionless_unscaled, \
                 cds.ppm, cds.ppm, cds.ppm, u.Rsun, u.Rsun, cds.K,\
                 u.dimensionless_unscaled ]
    elif category == 'backeb':
        colnames = ["tic", "ebNumber", "orbitalPeriod", "epoch", "b", "omega", \
                    "eccentricity", "1stStarDepth", "2ndStarDepth", "contactBinAmplitude", \
                    "1stRadiusStar", "2ndRadiusStar", "tempEff", "logg", "magOffset", "ra", "dec"]
        units = [u.dimensionless_unscaled, u.dimensionless_unscaled, u.day, u.day, \
                 u.dimensionless_unscaled, u.dimensionless_unscaled, u.dimensionless_unscaled, \
                 cds.ppm, cds.ppm, cds.ppm, u.Rsun, u.Rsun, cds.K,\
                 u.dimensionless_unscaled, u.mag, u.degree, u.degree]
    elif category == 'planet':
        colnames = ["tic", "planetNumber", "orbitalPeriod", "epoch", "radiusPlanetRatio", "radiusPlanet", "b", "a", "duration", "depth", "insolation", "radiusStar", "massStar", "tempEff", "logg", "metallicity"]
        units = [u.dimensionless_unscaled, u.dimensionless_unscaled, u.day, u.day, u.dimensionless_unscaled, u.Rearth, u.dimensionless_unscaled, u.hour, cds.ppm, u.dimensionless_unscaled, u.dimensionless_unscaled, u.Rsun, u.Msun, u.K, u.dimensionless_unscaled, u.dex]
    elif category == 'star':
        colnames = ["tic", "kic"]
        units = [u.dimensionless_unscaled, u.dimensionless_unscaled]
    
    targets = astab.Table.read(fname, format='ascii')
    
    assert len(targets.colnames) == len(colnames) 
    assert len(targets.colnames) == len(units)
    for colname, unit, colname_new in zip(targets.colnames, units, colnames):
        targets[colname].unit = unit
        targets.rename_column(colname, colname_new)
    
    return targets
    
def loadOnly(category, includeStarVar=True, dataDir='data/lcs', truthDir="data/truthFiles", 
             includeMultipleSystem=False, timeColumn="CADENCENO", fluxColumn="PDCSAP_FLUX", 
             fluxErrColumn=None, needle="tess*.fits", verbose=False):
    """
    returns a list of list of the targets in given category
    
    :param category: eb, backeb, planet or none
    
    """
    
    timeStart = datetime.now()
    
    truths = {}
    truthLists = ["eb", "backeb", "planet", "star"]
    catTruthList = []
    nonCatTruthList = []
    starVarList = []
    for cat in truthLists:
        tics = list(getTruthList(cat, truthDir)["tic"])
        truths[cat] = tics
        if cat == category:
            catTruthList = tics
        elif cat == "star":
            starVarList = tics
        else:
            nonCatTruthList += tics
        
    catTruthList = np.array(catTruthList)
    nonCatTruthList = np.array(nonCatTruthList)
    starVarList = np.array(starVarList)
    
    if not includeMultipleSystem:
        catTruthList, catTruthListCount = np.unique(catTruthList, return_counts=True)
        catTruthList = catTruthList[np.where(catTruthListCount == 1)]
    
    availableFiles = glob.glob(os.path.join(dataDir, needle))
    availableTics = []
    
    timestamps = []
    fluxstamps = []
    fluxerrstamps = []
    
    stepShout = 5
    lastShoutOut = -stepShout
    for ii_, fname in enumerate(availableFiles):
        
        if (ii_+1)/len(availableFiles) * 100 > lastShoutOut + stepShout:
            shoutOutval = (ii_+1)/len(availableFiles) * 100
            logger.info("Loading at {:2.0f}%".format(shoutOutval))
            lastShoutOut += stepShout
        
        tic = int(fname.split("-")[1])
        
        if tic in catTruthList or category is "none":
            logger.debug("TIC {} is in cat {}".format(tic, category))
            check = True
            include = True
        else:
            check = False
            include = False
        if tic in nonCatTruthList and check:
            logger.debug("TIC {} is in cat {}".format(tic, "NON"))
            include = False
        if tic in starVarList and check:
            logger.debug("TIC {} is in cat {}".format(tic, "STAR"))
            if not includeStarVar: 
                include = False 
                
        if include:
            availableTics.append(tic)
            tab = astab.Table.read(fname, format="fits")
            logger.debug("Adding TIC {} to available TICs".format(tic))
            timestamps += [list(tab[timeColumn])]
            fluxstamps += [list(tab[fluxColumn])]
            if fluxErrColumn is not None:
                fluxerrstamps += [list(tab[fluxErrColumn])]
                
    timestamps = np.array(timestamps).T
    fluxstamps = np.array(fluxstamps).T    
    if fluxErrColumn is not None:      
        fluxerrstamps = np.array(fluxerrstamps).T   
        assert np.shape(fluxerrstamps) == (len(tab[timeColumn]), len(availableTics))
        
    availableTics = np.array(availableTics)
    logger.info("Found {} lightcurves for cat {} out of {} files available".format(len(availableTics), category, len(availableFiles)))
    assert np.shape(timestamps) == (len(tab[timeColumn]), len(availableTics))
    assert np.shape(fluxstamps) == (len(tab[timeColumn]), len(availableTics))
    
    logger.info("Load function for target category '{}' took {} to complete".format(category,datetime.now()-timeStart))
    
    if fluxErrColumn is None:
        return availableTics, timestamps, fluxstamps
    else:
        return availableTics, timestamps, fluxstamps, fluxerrstamps
    
def fits2pickle(categories, saveDir="data/", includeStarVar=True, dataDir='data/lcs', truthDir="data/truthFiles", 
             includeMultipleSystem=False, timeColumn="CADENCENO", fluxColumn="PDCSAP_FLUX", 
             fluxErrColumn=None, needle="tess*.fits", verbose=False):
    """
    Load the LCs from different categories and save them to a pickle in saveDir.
    
    :param categories: what category of data to load, a list of string
    :param saveDir: where to save the pickle
    """
    
    for category in categories:
        loaded = loadOnly(category, includeStarVar, dataDir, truthDir, includeMultipleSystem, timeColumn, fluxColumn, fluxErrColumn, needle, verbose)
        if fluxErrColumn is None:
            availableTics, timestamps, fluxstamps = loaded
            fluxerrstamps = None
        else:
            availableTics, timestamps, fluxstamps, fluxerrstamps = loaded
        
        if (np.shape(np.unique(timestamps)) == (np.shape(timestamps)[0],)):
            timestamps = np.unique(timestamps)
        
        fname = os.path.join(saveDir, "{}.pkl".format(category))
        io.pickleWrite([availableTics, timestamps, fluxstamps, fluxerrstamps], fname)
        
        del availableTics, timestamps, fluxstamps, fluxerrstamps

def predictClass(pred, threshold):
    classification = np.zeros_like(pred)
    classification[pred >= threshold] = 1
    return classification

    