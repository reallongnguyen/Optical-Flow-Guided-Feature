import itertools

import numpy as np
import matplotlib.pyplot as plt

# from sklearn.metrics import confusion_matrix

UCF11_CLASSES_NAME = [
    'basketball',
    'biking',
    'diving',
    'g_swing',
    'h_riding',
    's_juggling',
    'swing',
    't_swing',
    't_jumping',
    'v_spiking',
    'walking',
]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = np.array([
    [33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 30, 0, 3, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 11, 0, 20, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 28, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 20, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0],
    [0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 19],
])

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=UCF11_CLASSES_NAME,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=UCF11_CLASSES_NAME, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
