import numpy as np
from keras import backend as K
import pandas as pd

import logging
logger = logging.getLogger(__name__)

def getIdsConfusionMatrix(predClass, labels):
    # Get the true positives and false negatives ids
    idsTP = np.where(np.logical_and(predClass == labels, labels==1))[0]
    idsFN = np.where(np.logical_and(predClass != labels, labels==1))[0]
    idsTN = np.where(np.logical_and(predClass == labels, labels==0))[0]
    idsFP = np.where(np.logical_and(predClass != labels, labels==0))[0]
    
    return idsTP, idsFN, idsTN, idsFP

class FiltersImages():
    """
    Follows https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html and
    https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py and
    https://fairyonice.github.io/Visualization%20of%20Filters%20with%20Keras.html
    """
    
    def __init__(self, model):
        
        self.model = model

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        self.layerDict = dict([(layer.name, layer) for layer in self.model.layers])
        
        # this is the placeholder for the input signal
        self.inputSignal = model.input
        
    def _normalize(self, x):
        """
        utility function to normalize a tensor by its L2 norm
        """
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
        
    def getLossFunctionFromFilter(self, layerName, filterIndex):
        """
        Build a loss function that maximizes the activation
        of the nth filter of the layer considered
        """
        layerOutput = self.layerDict[layerName].output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layerOutput[:, filterIndex, :])
        else:
            loss = K.mean(layerOutput[:, :, filterIndex])
    
        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, self.inputSignal)[0]
    
        # normalization trick: we normalize the gradient
        grads = self._normalize(grads)
    
        # this function returns the loss and grads given the input picture
        return K.function([self.inputSignal], [loss, grads])

    def gradientAscent(self, computeLoss, signalLength, signalDimension, nstep=20, step=1.):
         
        # we start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            filterImage = np.random.random((1, signalLength, signalDimension))
        else:
            filterImage = np.random.random((1, signalDimension, signalLength))
            
        #input_img_data = (input_img_data - 0.5) * 20 + 128
        
        # we run gradient ascent for 20 steps
        for ii in range(nstep):
            
            lossValue, gradsValue = computeLoss([filterImage])
            filterImage += gradsValue * step
            logger.debug("Step {}/{}, current loss = {}".format(ii+1, nstep, lossValue))

            if lossValue <= 0.:
                # some filters get stuck to 0, we can skip them
                break
        return lossValue, filterImage
    
    def getFiltersFromLayer(self, layerName):
        
        filtersN = self.layerDict[layerName].filters
        filtersList = []
        filtersLosses = []
        
        _, signalDimension, signalLength = self.layerDict[layerName].input.shape
        
        for ii, filterIndex in enumerate(range(filtersN)):
            logger.info("Finding the maximal activation signal for filter {}/{}".format(ii+1, filtersN))
            computeLoss = self.getLossFunctionFromFilter(layerName, filterIndex)
            lossValue, filterImage = self.gradientAscent(computeLoss, signalDimension=signalDimension, signalLength=signalLength)
            
            if signalDimension == 1:
                filterImage = filterImage.reshape(signalLength)

            filtersLosses.append(lossValue)
            filtersList.append(filterImage)
        
        filtersLosses = np.array(filtersLosses)
        filtersList = np.array(filtersList)
        
        return filtersLosses, filtersList
    
def classificationReport2csv(report, filepath):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split(' ')
        row_data = list(filter(None, row_data))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filepath, index = False)

        