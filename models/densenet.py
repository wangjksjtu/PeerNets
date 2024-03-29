### some bugs to fix !!!

from tensorflow.python.keras._impl.keras.layers import (Activation,
                                                        BatchNormalization,
                                                        Convolution2D, Dense, Dropout,
                                                        Flatten, AveragePooling2D,
                                                        GlobalAveragePooling2D,
                                                        Input, MaxPooling2D,
                                                        add, concatenate)
from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.optimizers import SGD
from tensorflow.python.keras._impl.keras.regularizers import l2
import keras.backend as K

'''
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.optimizers import SGD
'''
'''
Based on the implementation here : https://github.com/Lasagne/Recipes/blob/master/papers/densenet/densenet_fast.py
'''

def conv_block(ip, nb_filter, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 3x3, Conv2D, optional dropout

    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with batch_norm, relu and convolution2d added

    '''

    x = Activation('relu')(ip)
    x = Convolution2D(nb_filter, 3, 3, padding="same", use_bias=False,
                    kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(ip, nb_filter, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional dropout and Maxpooling2D

    Args:
        ip: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = Convolution2D(nb_filter, 1, 1, padding="same", use_bias=False,
                    kernel_regularizer=l2(weight_decay))(ip)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                        beta_regularizer=l2(weight_decay))(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones

    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with nb_layers of conv_block appended

    '''

    # concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    ### concatenate bug here ###
    concat_feat = x

    for i in range(nb_layers):
        x = conv_block(concat_feat, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=3, name='concat_' + str(i))
        nb_filter += growth_rate

    return concat_feat, nb_filter


def create_dense_net(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                    weight_decay=1E-4, verbose=True):
    ''' Build the create_dense_net model

    Args:
        nb_classes: number of classes
        img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay

    Returns: keras tensor with nb_layers of conv_block appended

    '''

    model_input = Input(shape=img_dim, name="img_input")

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Convolution2D(nb_filter, 3, 3, padding="same", name="initial_conv2D", use_bias=False,
                    kernel_regularizer=l2(weight_decay))(model_input)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                            beta_regularizer=l2(weight_decay))(x)


    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                            weight_decay=weight_decay)

    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    densenet = Model(input=model_input, output=x, name="create_dense_net")

    if verbose: print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet

def build_model_densenet(lr, decay, classes=10):
    model = create_dense_net(classes, (32,32,3), depth=40, growth_rate=12)
    sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def get_model(nb_classes):
    model = create_dense_net(nb_classes, (32,32,3), depth=40, growth_rate=12)
    return model