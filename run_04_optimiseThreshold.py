from datetime import datetime
import sklearn.metrics as metrics 
import matplotlib.pyplot as plt
import os
import logging
import numpy as np
import shutil

import tmllc

logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.INFO)

runName = "fakeWideParams"
modelName = "convNetLSTM"
metricFunction = metrics.accuracy_score
metric = metricFunction.__name__
dataDir = os.path.join("runs", runName, "data")
runDir = os.path.join("runs", runName, modelName)
saveDirDiag = os.path.join(runDir, "diagnostics")
np.random.seed(10)

figSave = True
figShow = True

if os.path.exists(saveDirDiag) and figSave:
    logging.warning("There exists already a diagnostic directory. Removing it. You'll have to rerun the evalution of the model")
    shutil.rmtree(saveDirDiag)

if not os.path.exists(saveDirDiag) and figSave:
    os.makedirs(saveDirDiag)

timeStart = datetime.now()

validFeatures = tmllc.io.loadDataset("validFeatures", dataDir)
validLabels = tmllc.io.loadDataset("validLabels", dataDir)

model = tmllc.io.loadModel(runDir, name="model")
preds = model.predict(validFeatures)

# Compute ROC curve and area the curve
fpr, tpr, thresholds = metrics.roc_curve(validLabels, preds)
roc_auc = metrics.auc(fpr, tpr)

metricValues = []
for threshold in thresholds:
    predClass = tmllc.utils.predictClass(preds, threshold)
    metricValue = metricFunction(validLabels, predClass)
    metricValues.append(metricValue)
metricValues = np.array(metricValues)
bestThresholdId = np.argmax(metricValues)
bestThreshold = thresholds[bestThresholdId]

data = {"threshold": float(bestThreshold), "criterion": metric, "utcdate": "{}".format(datetime.utcnow().isoformat())}
tmllc.io.jsonWrite(data, os.path.join(runDir, "threshold.json"))
logging.critical("Best threshold is chosen to be {}".format(bestThreshold))

fig = plt.figure()
plt.scatter(fpr[bestThresholdId], tpr[bestThresholdId], color="red", label="Performance at threshold", zorder=99)
plt.plot(fpr, tpr, label='ROC (AUC = %0.4f)' % (roc_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([-0.01,1.01])
plt.legend(loc=0)
plt.grid()
plt.tight_layout()
if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, "roc"), fig)

if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, "roc"), fig)

fig = plt.figure()
plt.plot(thresholds, metricValues)
plt.xlabel('Threshold')
plt.ylabel('{} score'.format(metric))
plt.xlim([-0.01,1.01])
plt.axvline(bestThreshold, c="red")
plt.grid()
if figSave: tmllc.plots.figSave(os.path.join(saveDirDiag, "{}".format(metric)), fig)
    
if figShow: plt.show()

logging.info("Optimising the threshold of the model took {}".format(datetime.now()-timeStart))
