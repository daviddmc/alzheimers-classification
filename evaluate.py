from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os 


def evaluate_model(m, test_data, test_labels, opts):    
    # load trained weight
    m.load_weights(os.path.join(opts['output_path'], opts['exp_name'], opts['epoch_weights_file']))
    # predict     
    prediction = m.predict(test_data)
    if opts['task'] == 'binary':
        prediction_labels = prediction > 0.5
    else:
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