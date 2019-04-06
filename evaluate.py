#import matplotlib.pyplot as plt
#import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os 

'''
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
'''

def evaluate_model(m, test_data, test_labels, opts):    
    # load trained weight
    m.load_weights(os.path.join(opts['output_path'], opts['exp_name'], opts['epoch_weights_file']))
    # predict     
    prediction = m.predict(test_data)
    prediction_labels = np.argmax(prediction, axis=1)
    test_labels = np.squeeze(test_labels)
    prediction_labels = np.squeeze(prediction_labels)
    # calculate test acc
    print('Accuracy on test data:', np.mean(test_labels == prediction_labels))
    print('Classification Report')
    print(classification_report(test_labels, prediction_labels, target_names = opts['classes']))
    print('confusion matrix')
    cnf_matrix = confusion_matrix(test_labels, prediction_labels)
    print(cnf_matrix)
    #np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes = class_names, normalize=False, title='Confusion matrix')
    #plt.show()