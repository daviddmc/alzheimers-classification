from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import os
import json


def save_model_and_weights(model, model_file, weight_file):
    model.save(model_file)
    model.save_weights(weight_file)
    print('model and weights saved')

def save_model_history(m, history_file):
    with open(history_file, 'w') as history_json_file:
        json.dump(m.history, history_json_file)
    print('model history saved')
    
def visualise_accuracy(m):
    plt.plot(m.history['acc'])
    plt.plot(m.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    

def visualise_loss(m):
    plt.plot(m.history['loss'])
    plt.plot(m.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    
def model_callbacks(epoch_weights_file, early_stopping_patience):
    checkpoint = ModelCheckpoint(epoch_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stopping_patience, verbose=1, mode='auto')
    return [checkpoint, early_stopping]


def fit_model(model, train_data, train_labels, val_data, val_labels, opts):
    output_path = os.path.join(opts['output_path'], opts['exp_name'])
    model_weights_file = os.path.join(output_path, opts['model_weights_file'])
    epoch_weights_file = os.path.join(output_path, opts['epoch_weights_file'])
    model_file = os.path.join(output_path, opts['model_file'])
    history_file = os.path.join(output_path, opts['history_file'])
        
    callbacks_list = model_callbacks(epoch_weights_file, opts['early_stopping_patience'])

    m = model.fit(train_data, train_labels, 
                  batch_size=opts['batch_size'], epochs=opts['epochs'], 
                  verbose=1, shuffle=True, 
                  validation_data=(val_data,val_labels), callbacks=callbacks_list)
    
    save_model_and_weights(model, model_file, model_weights_file)
    save_model_history(m, history_file)
    
    #visualise_accuracy(m)
    #visualise_loss(m)
    
    return m