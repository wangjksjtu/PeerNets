import os
import sys

'''
import tensorflow as tf
from tensorflow.python.keras._impl.keras.layers import (Activation, Conv2D,
                                                        Dropout,
                                                        GlobalAveragePooling2D,
                                                        Input)
from tensorflow.python.keras._impl.keras.models import Sequential
'''
from keras.layers import Activation, Conv2D, Dropout, GlobalAveragePooling2D
from keras.models import Sequential

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from pr_layer import PeerRegularization, PeerRegularization_slow


def get_model(nb_classes=10, add_peer=True):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding = 'same', input_shape=(32, 32, 3), name="img"))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', strides = (2,2)))
    model.add(Dropout(0.5))

    if add_peer:
        model.add(PeerRegularization(5))

    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', strides = (2,2)))
    model.add(Dropout(0.5))

    if add_peer:
        model.add(PeerRegularization(5))

    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_classes, (1, 1), padding='valid'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    return model

'''
def get_model(nb_classes=10):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding = 'same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_classes, (1, 1), padding='valid'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    return model
'''
'''
def get_model_tf(net, bn_decay=None, nb_classes=10):
    net = tf_util.conv2d(net, 32, [3,3],
                         padding='SAME', stride=[1,1],
                         bn=False,
                         scope='conv0', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [3,3],
                         padding='SAME', stride=[1,1],
                         bn=False,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 32, [3,3],
                         padding='SAME', stride=[2,2],
                         bn=False,
                         activation_fn=None,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.dropout(net, is_training=None, keep_prob=0.5,
                          scope='dp1')

    net = tf_util.conv2d(net, 128, [3,3],
                         padding='SAME', stride=[1,1],
                         bn=False,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [3,3],
                         padding='SAME', stride=[1,1],
                         bn=False,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [3,3],
                         padding='SAME', stride=[2,2],
                         bn=False,
                         activation_fn=None,
                         scope='conv5', bn_decay=bn_decay)
    net = tf_util.dropout(net, is_training=None, keep_prob=0.5,
                          scope='dp2')

    net = tf_util.conv2d(net, 128, [3,3],
                         padding='SAME', stride=[1,1],
                         bn=False,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, nb_classes, [3,3],
                         padding='VALID', stride=[1,1],
                         bn=False,
                         activation_fn=None,
                         scope='conv8', bn_decay=bn_decay)

    net = tf.nn.avg_pool(net, ksize=[1, 6, 6, 1], strides=[1, 1, 1, 1], padding='VALID')
    print net
    net = tf.reduce_mean(net, axis=[1, 2])
    print net
'''
