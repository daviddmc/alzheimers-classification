import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Convolution1D, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, ZeroPadding2D, BatchNormalization, ReLU, Dropout, Activation, Convolution3D, MaxPooling3D, Input
from tensorflow.keras.models import Model, Sequential

def ACNN(x, nfeat, nfeat_fac, depth, fc, dropout_rate, l2_weight, **kwarg):
    
    for i in range(depth):
        x = Convolution3D(nfeat, (3, 3, 3), kernel_regularizer = tf.keras.regularizers.l2(l2_weight))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        nfeat *= nfeat_fac
        
    x = Flatten()(x)
    
    for n_unit in fc:
        x = Dense(n_unit, activation='relu', kernel_regularizer= tf.keras.regularizers.l2(l2_weight))(x)
        x = Dropout(dropout_rate)(x)
        
    return x


def build_cnn(opts):
    print('build CNN')

    input_node = Input(opts['input_shape'])
    x = globals()[opts['model']['name']](input_node, **opts['model'])
    if opts['task'] == 'binary':
        output_node = Dense(1, activation='sigmoid')(x)
    else:
        output_node = Dense(opts['nClasses'], activation='softmax')(x)
    model = Model(inputs=input_node, outputs=output_node)
    
    print('output shape')
    for layer in model.layers:
        print(layer.output_shape)
        
    kwarg = opts['optimizer']
    opt = getattr(tf.keras.optimizers, opts['optimizer']['name'])(**{k:kwarg[k] for k in kwarg if k!='name'})
    
    if opts['task'] == 'binary':
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    return model