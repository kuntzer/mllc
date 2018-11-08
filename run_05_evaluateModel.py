import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import scipy.stats as stats

import tmllc

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

runName = "fakeWideParams"
modelName = "LSTMOnly"
runDir = os.path.join("runs", runName, modelName)
saveDirDiag = os.path.join(runDir, "diagnostics")
saveDirData = os.path.join("runs", runName, "data")

falsePositiveQQPlot = False
filtersCompute = False
figSave = True
figShow = False

threshold = tmllc.io.jsonRead(os.path.join(runDir, "threshold.json"))
threshold = threshold['threshold']

model = tmllc.io.loadModel(runDir, name="model")
print(model.summary())

if not os.path.exists(saveDirDiag) and figSave:
    os.mkdir(saveDirDiag)

#--------------------------------------------------------------------------------------------------
# Plot the images that maximise the filters
if filtersCompute:
    layerName = "conv1d_1"
    filters = tmllc.diagnostics.FiltersImages(model)
    filtersLosses, filtersList = filters.getFiltersFromLayer(layerName)
    
    fig = plt.figure()
    for ii, (loss, filterImage) in enumerate(zip(filtersLosses, filtersList)):
        plt.plot(filterImage, label="{}: {:1.2f}".format(ii+1, loss))
    plt.legend(loc='upper right')
    
    if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, "filtersImage"), fig)


#-----------------------------------------------------------------------------------
# Plot the kernels
if filtersCompute:
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    kernels = layer_dict["conv1d_1"]
    filters, biases = kernels.get_weights()
    filters = filters[:,0,:]
    fig = plt.figure()
    for ii in range(kernels.filters):
        plt.plot(filters[:,ii] + biases[ii])
        
    if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, "filtersKernels"), fig)

if figShow and filtersCompute: plt.show()

#-----------------------------------------------------------------------------------
history = tmllc.io.loadHistory(runDir, name="model")
# summarize history for the metric in the history file

# Find out what metric we used during the training of the model
keys = list(history.keys())
keysLosses = [k for k in keys if "loss" in k]
[keys.remove(k) for k in keysLosses]

fig = plt.figure(figsize=(8,12))
# summarize history for metric
ax1 = fig.add_subplot(211)

keysVal = [k for k in keys if "val_" in k]
assert len(keysVal) == 1
keysVal = keysVal[0]
keys.remove(keysVal)
assert len(keys) == 1
keys = keys[0]

if keys == 'acc':
    ylabel = 'Accuracy'
else:
    raise NotImplementedError("metric {} is unknown".format(ylabel))

plt.plot(history[keys])
plt.plot(history[keysVal])
plt.ylabel(ylabel)
plt.legend(['Training', 'Validation'], loc='upper left')
plt.grid()

# summarize history for loss
ax2 = fig.add_subplot(212, sharex=ax1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.tight_layout()
if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, "metricsHistories"), fig)

#--------------------------------------------------------------------------------------------------
for dataset in ["train", "valid"]:
    
    msg = "Treating {} dataset".format(dataset.upper())
    lenast = len(msg) + 12
    msg1 = "{} {} {}".format("*" * 5, msg, "*" * 5)
    msg2 = lenast * "*"
    logging.info(msg2)
    logging.info(msg1)
    logging.info(msg2)
    
    features = tmllc.io.loadDataset("{}Features".format(dataset), saveDirData)
    labels = tmllc.io.loadDataset("{}Labels".format(dataset), saveDirData)
    truthTable = tmllc.io.loadTable("{}TruthTable".format(dataset), saveDirData)

    preds = model.predict(features)
    preds = preds.reshape(np.size(preds))
    
    predClass = tmllc.utils.predictClass(preds, threshold)

    #-----------------------------------------------------------------------------------
    # Confution Matrix and Classification Report
    
    print('Confusion Matrix')
    confusionMatrix = confusion_matrix(labels, predClass)
    print(confusionMatrix)
    print('Classification Report')
    LabelNames = ['Noise', 'Transit']
    classificationReport = classification_report(labels, predClass, target_names=LabelNames)
    print(classificationReport)
    tmllc.diagnostics.classificationReport2csv(classificationReport, os.path.join(saveDirDiag, "{}ClassificationReport.csv".format(dataset)))
    
    
    fig = tmllc.plots.confusionMatrix(confusionMatrix, LabelNames, normalize=False)
    if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, "{}ConfusionTable".format(dataset)), fig)
    
    fig = tmllc.plots.confusionMatrix(confusionMatrix, LabelNames, normalize=True)
    if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, "{}ConfusionTableNormalised".format(dataset)), fig)
    
    #-----------------------------------------------------------------------------------
    # Distribution plots of the false negatives (and true positives as well)
    
    idsTP, idsFN, idsTN, idsFP = tmllc.diagnostics.getIdsConfusionMatrix(predClass, labels)
    #"""
    plt.figure()
    bins = np.linspace(0,1, 101)
    plt.hist(preds[idsTP], bins=bins, color="green", alpha=0.5, label="True Pos")
    plt.hist(preds[idsFN], bins=bins, color="red", alpha=0.5, label="False Neg")
    plt.hist(preds[idsTN], bins=bins, color="blue", alpha=0.5, label="True Neg")
    plt.hist(preds[idsFP], bins=bins, color="yellow", alpha=0.5, label="False Pos")
    plt.axvline(threshold, c='k', ls='--', label="Threshold")
    plt.xlabel("Prediction")
    plt.legend(loc='best')
    plt.yscale('log', nonposy='clip')
    plt.xlim([0, 1])
    
    for coln in truthTable.colnames:
        fig = plt.figure()
        col = truthTable[coln]
        colT = col[np.hstack([idsTP, idsFN])]
        bins = np.linspace(np.amin(colT), np.amax(colT), 101)
        plt.hist(col[idsTP], bins=bins, color="green", alpha=0.5, label="True Pos")
        plt.hist(col[idsFN], bins=bins, color="red", alpha=0.5, label="False Neg")
        plt.title("Param: {}".format(coln))
        plt.legend(loc=0)
        if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, "{}FN{}".format(dataset, coln)), fig)
        if figShow: plt.show()
    
    #-----------------------------------------------------------------------------------
    # Diagnositic fo the false positives QQ plots
    if falsePositiveQQPlot:
        qqPlotDir = os.path.join(saveDirDiag, "QQplots", dataset)
        if not os.path.exists(qqPlotDir):
            os.makedirs(qqPlotDir)
            
        for idFP in idsFP:
            
            fp = features[idFP][0]
            fig = plt.figure(figsize=(6,10))
            ax = fig.add_subplot(211)
            res = stats.probplot(fp, plot=ax)
            
            ax = fig.add_subplot(212)
            plt.title(preds[idFP])
            plt.scatter(np.arange(len(fp)), fp)
            
            if figSave: tmllc.plots.figSave(os.path.join(qqPlotDir, "id{:06d}".format(idFP)), fig)
            if figShow: plt.show()
    
    #-----------------------------------------------------------------------------------

