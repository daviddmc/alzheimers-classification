from models import build_cnn
from train import fit_model
from dataset import load_dataset
from evaluate import evaluate_model
from options import opts

if __name__ == '__main__':

    # read dataset
    train_data, val_data, test_data, train_label, val_label, test_label = load_dataset(opts)
    
    # training
    if opts['run'] == 'train' or opts['run'] == 'both':
        # build model
        model = build_cnn(opts)
        # fit model
        fit_model(model, train_data, train_label, val_data, val_label, opts)
    
    # testing
    if opts['run'] == 'test' or opts['run'] == 'both':
        if opts['run'] == 'test':
            model = build_cnn(opts)
        # test
        evaluate_model(model, test_data, test_label, opts)