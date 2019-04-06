import os
import pickle


opts = {}

''' general '''

# name of experiment
opts['exp_name'] = 'ResNet_v1'

# train, test or both
opts['run'] = 'both'

''' data '''

# binary or ternary
opts['task'] = 'ternary'

if opts['task'] == 'binary':
    opts['classes'] = ["CN", "MCI+AD"]
elif opts['task'] == 'ternary':
    opts['classes'] = ["CN", "MCI", "AD"]
else:
    raise Exception('task error')

opts['nClasses'] = len(opts['classes'])
opts['input_shape'] = (62, 96, 96, 1)


''' I/O '''

opts['data_path'] = '../data'
opts['output_path'] = '../output'
opts['model_weights_file'] = 'weights_final.h5'
opts['epoch_weights_file'] = 'weights_best.hdf5'
opts['model_file'] = 'model.h5'
opts['history_file'] = 'history.json'

''' model '''

model = {}
model['name'] = 'ResNet'

if model['name'] == 'ACNN':
    model['l2_weight'] = 0.0001
    model['nfeat'] = 8
    model['nfeat_fac'] = 1
    model['depth'] = 3
    model['dropout_rate'] = 0.5
    model['fc'] = [2000, 500]
elif model['name'] == 'ResNet':
    model['block_fn'] = 'basic_block'
    model['repetitions'] = [2, 2, 2]
    model['k_first'] = 5
    model['nfeat'] = 16
else:
    raise Exception('model name error')
    
opts['model'] = model

''' train '''

opts['batch_size'] = 32 #64
opts['early_stopping_patience'] = 20
opts['epochs'] = 100

optimizer = {}
optimizer['name'] = 'Adadelta'

if optimizer['name'] == 'Adadelta':
    optimizer['lr'] = 1.0
    optimizer['rho'] = 0.95
    optimizer['epsilon'] = None
    optimizer['decay'] = 0.0
    optimizer['clipnorm'] = 1.
elif optimizer['name'] == 'Adam':
    optimizer['lr'] = 0.001,
    optimizer['beta_1'] = 0.9,
    optimizer['beta_2'] = 0.999,
    optimizer['epsilon'] = None
    optimizer['decay'] = 0.0
    optimizer['amsgrad'] = False
else:
    raise Exception('optimizer name error')
    
opts['optimizer'] = optimizer

#################### setup ###########################

output_path = os.path.join(opts['output_path'], opts['exp_name'])
if not os.path.isdir(output_path):
    os.makedirs(output_path)
    
if opts['run'] == 'test':
    with open(os.path.join(output_path, 'opts.pkl'), 'rb') as f:
        loaded_opts = pickle.load(f)
        loaded_opts['run'] = 'test'
        opts = loaded_opts
else:
    with open(os.path.join(output_path, 'opts.pkl'), 'wb') as f:
        pickle.dump(opts, f, pickle.HIGHEST_PROTOCOL)
