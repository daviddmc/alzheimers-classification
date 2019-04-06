import numpy as np
import tensorflow as tf
import os
import pandas as pd

def load_img(data_path, input_shape, set_name):
    filenames = [os.path.join(data_path, f) for f in sorted(os.listdir(data_path)) if set_name in f]
    data = []
    for f in filenames:
        print(f)
        data.append(np.load(f))
    data = np.concatenate(data)
    data = np.reshape(data, (-1,) + input_shape)
    data = normalize_0_1(data)
    return data

def normalize_0_1(data):
    for i in range(len(data)):
        data[i] /= np.amax(data[i])
    return data

def load_label(filename, train_valid_test, is_binary):
    label = pd.read_csv(filename)
    label = label.loc[label['train_valid_test'] == train_valid_test]
    label = np.asarray(label.diagnosis)     
    label = label.reshape((-1, 1)) - 1
    label = label.astype(np.uint8)
    if is_binary:
        label[label == 2] = 1
    return label
    
def load_dataset(opts):
    data_path = opts['data_path']
    input_shape = opts['input_shape']
    
    train_data = load_img(data_path,input_shape, 'train')
    val_data = load_img(data_path,input_shape, 'valid')
    test_data = load_img(data_path, input_shape,'test')
    
    label_name = 'adni_demographic_master_kaggle.csv'
    is_binary = opts['task'] == 'binary'
    train_label = load_label(os.path.join(data_path, label_name), 0, is_binary)
    val_label = load_label(os.path.join(data_path, label_name), 1, is_binary)
    test_label = load_label(os.path.join(data_path, label_name), 2, is_binary)
    
    print('train_data: ' + str(train_data.shape))
    print('val_data: ' + str(val_data.shape))
    print('test_data: ' + str(test_data.shape))
    
    return train_data, val_data, test_data, train_label, val_label, test_label