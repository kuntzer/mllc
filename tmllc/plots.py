import numpy as np
import matplotlib.pyplot as plt
import itertools
import subprocess
import os

def confusionMatrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.get_cmap("Blues")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title is not None: plt.title(title)
    
    plt.colorbar()
    tick_marks = np.arange(len(classes))


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(tick_marks, classes)#, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    
    return fig

def figSave(filename, fig, pdf=True, pdfTransparence=True, dpi=300):
    """
    Saves a figure to the disc
    
    :param filename: the name of the file (! without extension)
    :param pdf: if `True` also saves a cropped pdf
    :param pdfTransparence: does the pdf have transparency
    :param dpi: dpi for png file
    
    .. warning:: the cropped pdf function might only work on unix systems
    """

    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig.savefig(filename+'.png', dpi=dpi)

    if pdf: 
        fig.savefig(filename+'.pdf',transparent=pdfTransparence)
        command = 'pdfcrop {}.pdf'.format(filename)
        subprocess.check_output(command, shell=True)
        os.system('mv {}-crop.pdf {}.pdf'.format(filename, filename))
