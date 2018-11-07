import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import logging
import glob
import os

import tmllc

logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.INFO)

runName = "fakeDataLarge"
saveDirDiag = os.path.join("runs", runName, "comparison")
figSave = True
figShow = True

pattern = os.path.join("runs", runName, "*", "modelDefinition.json")
models = glob.glob(pattern)

dataset = "valid"
saveDirData = os.path.join("runs", runName, "data")
features = tmllc.io.loadDataset("{}Features".format(dataset), saveDirData)
labels = tmllc.io.loadDataset("{}Labels".format(dataset), saveDirData)
truthTable = tmllc.io.loadTable("{}TruthTable".format(dataset), saveDirData)

modelNames = []
f1s = []
accs = []
precisions = []
recalls = []
ROCfprs = []
ROCtprs = []
ROCaucs = []
lossHistory = []
lossValHistory = []

if not os.path.exists(saveDirDiag) and figSave:
    os.mkdir(saveDirDiag)

for modelfname in models:
    
    path = modelfname.split("/")
    modelPath = os.path.join(*path[:-1])
    modelName = path[-2]
    modelNames.append(modelName)
    
    logging.info("Working on model {}".format(modelName))
    
    model = tmllc.io.loadModel(modelPath)
    
    threshold = tmllc.io.jsonRead(os.path.join(modelPath, "threshold.json"))
    threshold = threshold['threshold']
    
    history = tmllc.io.loadHistory(modelPath)
    lossValHistory.append(history['val_loss'])
    lossHistory.append(history['loss'])
    
    preds = model.predict(features)
    preds = preds.reshape(np.size(preds))
    
    predClass = tmllc.utils.predictClass(preds, threshold)
    
    f1 = metrics.f1_score(labels, predClass)
    f1s.append(f1)
    acc = metrics.accuracy_score(labels, predClass)
    accs.append(acc)
    precision = metrics.precision_score(labels, predClass)
    precisions.append(precision)
    recall = metrics.recall_score(labels, predClass)
    recalls.append(recall)
    
    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    ROCfprs.append(fpr)
    ROCtprs.append(tpr)
    ROCaucs.append(metrics.auc(fpr, tpr))

ind = np.arange(len(modelNames))

metricsLabels = ["$F_1$ score", "Precision", "Recall", "Accuracy"]
fnames = ["scoreF1", "scorePrecision", "scoreRecall", "scoreAccuracy"]
for arr, metricLabel, fname in zip([f1s, precisions, recalls, accs], metricsLabels, fnames):
    fig, ax = plt.subplots()
    plt.grid(True)
    bars = plt.bar(ind, arr, zorder=9)
    bestId = np.argmax(arr)
    bars[bestId].set_facecolor("r")
    ax.set_xticks(ind)
    ax.set_xticklabels(modelNames)
    plt.title(metricLabel)
    
    if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, fname), fig)
    
fig = plt.figure()
for fpr, tpr, auc, modelName in zip(ROCfprs, ROCtprs, ROCaucs, modelNames):
    plt.plot(fpr, tpr, label='{} (AUC = {:0.4f})'.format(modelName, auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.01,1.01])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, "roc"), fig)

fig = plt.figure()
for losses, name in zip([lossHistory, lossValHistory], ["Training loss", "Validation loss"]):
    
    if name == "Training loss":
        ls = '--'
        tr = True
        trColors = []
    else:
        ls = '-'
        tr = False
        
    for ii, (loss, modelName) in enumerate(zip(losses, modelNames)):
        
        if tr:
            color = None
            label = None
        else:
            color = trColors[ii]
            label = modelName
        
        plot = plt.plot(loss, label=label, ls=ls, color=color)
        if tr:
            trColors.append(plot[0].get_color())

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.grid(True)
if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, "loss"), fig)

if figShow: plt.show()
