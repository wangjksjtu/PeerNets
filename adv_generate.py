import logging
import os

import numpy as np
from scipy.misc import imsave

import h5py
import provider
import tensorflow as tf
from cleverhans.attacks import (LBFGS, CarliniWagnerL2, DeepFool,
                                FastGradientMethod, SaliencyMapMethod)
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_eval, model_train
from keras.datasets import cifar10
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D)
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils, plot_model
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def data_cifar10():
    """
    Preprocess CIFAR10 dataset
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def build_model(lr, decay, mlps=[32, 64, 128, 64, 32], classes=10):
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32)))
    for out_dim in mlps:
        model.add(Dense(out_dim, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(classes, activation=None))

    model.add(Activation('softmax'))
    sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # print (model.count_params())
    # model.summary()
    return model


def build_model_cifar(lr, decay):
    # all-cnns
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), padding = 'same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (3, 3), padding='same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (3, 3), padding='same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(10, (1, 1), padding='valid'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def adv_generate(nb_epochs=25, batch_size=128,
                 learning_rate=0.001,
                 clean_train=True,
                 testing=False,
                 nb_filters=64, num_threads=None,
                 data='cifar', adv_attack='fgsm', 
                 save_dir='data'):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    # set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    config = tf.ConfigProto(**config_args)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    if data == "mnist":
        # Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                      train_end=60000,
                                                      test_start=0,
                                                      test_end=10000)
    else:
        X_train, Y_train, X_test, Y_test = data_cifar10()

    # print (Y_test.shape)
    '''
    for i in range(Y_test.shape[0]):
        img = np.squeeze(X_test[i,:,:,:])
        imsave(os.path.join("benign", str(i) + ".jpg"), img)

    for i in range(Y_test.shape[0]):
        img = np.squeeze(X_test[i,:,:,:])
        benign_path = "benign_" + str(np.argmax(Y_test[i,:], axis=0))
        if not os.path.exists(benign_path):
        	os.makedirs(benign_path)
        imsave(os.path.join(benign_path, str(i) + ".jpg"), img)
    '''
    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    if data == 'mnist':
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    else:
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    rng = np.random.RandomState([2018, 7, 18])

    if clean_train:
        if data == 'mnist':
            model = build_model(0.01, 1e-6)
        else:
            model = build_model_cifar(0.01, 1e-6)

        preds = model(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == 10000, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)
        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                    args=train_params, rng=rng)

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc


        if adv_attack == "FGSM":
        # Initialize the attack object and graph
            # FGSM
            print "FGSM ATTACK..."
            fgsm_params = {'eps': 0.1,
                           'clip_min': 0.,
                           'clip_max': 1.}
            fgsm = FastGradientMethod(model, sess=sess)
            adv_x = fgsm.generate(x, **fgsm_params)
            preds_adv = model(adv_x)
        elif adv_attack == "CWL2":
            # CWL2
            print "CWL2 ATTACK..."
            cwl2_params = {'batch_size': 8}
            cwl2 = CarliniWagnerL2(model, sess=sess)
            adv_x = cwl2.generate(x, **cwl2_params)
            preds_adv = model(adv_x)
        elif adv_attack == "JSMA":
            # JSMA
            print "JSMA ATTACK..."
            jsma = SaliencyMapMethod(model, back='tf', sess=sess)
            jsma_params = {'theta': 1., 'gamma': 0.1,
                           'clip_min': 0., 'clip_max': 1.}
            adv_x = jsma.generate(x, **jsma_params)
            preds_adv = model(adv_x)
        elif adv_attack == "DeepFool":
            # DeepFool
            print "DeepFool ATTACK..."
            deepfool = DeepFool(model, sess=sess)
            deepfool_params = {'nb_candidate':10, 'overshoot':0.02, 'max_iter':50,
                               'clip_min':0.0, 'clip_max':1.0}
            adv_x = deepfool.generate(x, **deepfool_params)
            preds_adv = model(adv_x)
        elif adv_attack == "LBFGS":
            # LBFGS
            print "LBFGS ATTACK..."
            lbfgs_params = {'y_target': y, 'batch_size':100}
            lbfgs = LBFGS(model, sess=sess)
            adv_x = lbfgs.generate(x, **lbfgs_params)
            preds_adv = model(adv_x)
        
        
        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        adv_imgs = []
        adv_imgs_test = []

        if not adv_attack == "LBFGS":
            for i in range(5000):
                adv_imgs_train, _ = sess.run([adv_x, preds_adv], feed_dict={x: X_train[i*10:(i+1)*10]})
                adv_imgs.append(adv_imgs_train)
            adv_imgs = np.vstack(adv_imgs)
            print (adv_imgs.shape)
            for i in range(1000):
                adv_imgs_tmp, _ = sess.run([adv_x, preds_adv], feed_dict={x: X_test[i*10: (i+1)*10]})
                adv_imgs_test.append(adv_imgs_tmp)
            adv_imgs_test = np.vstack(adv_imgs_test)
        else:
            for i in range(500):
                target = np_utils.to_categorical((np.argmax(Y_train[i*100: (i+1)*100], axis = 1) + 1) % 10, 10)
                adv_imgs_train, _ = sess.run([adv_x, preds_adv], feed_dict={x: X_train[i*100: (i+1)*100],
                                                                            y: target})
                print ('train image: %s' % str(i))
                adv_imgs.append(adv_imgs_train)
            print (adv_imgs.shape)

            for i in range(100):
                target = np_utils.to_categorical((np.argmax(Y_train[i*100: (i+1)*100], axis = 1) + 1) % 10, 10)
                adv_imgs_train, _ = sess.run([adv_x, preds_adv], feed_dict={x: X_train[i*100: (i+1)*100],
                                                                            y: target})
                adv_imgs_test.append(adv_imgs_tmp)
                print ('test image: %s' % str(i))
            adv_imgs_test = np.vstack(adv_imgs_test)

        '''
        for i in range(6):
            target = np_utils.to_categorical((np.argmax(Y_train[i*10000: (i+1)*10000, ...], axis = 1) + 1) % 10, 10)
            adv_imgs_train, adv_labels_train = sess.run([adv_x, preds_adv], feed_dict={x: X_train[i*10000: (i+1)*10000,...],
                                                                                       y: target})
        for i in range(60000):
            target = np_utils.to_categorical((np.argmax(Y_train[i:i+1, ...], axis = 1) + 1) % 10, 10)
            adv_imgs_train = sess.run([adv_x], feed_dict={x: X_train[i:i+1,...], y: target})
            print (len(adv_imgs_train), adv_imgs_train[0].shape, adv_imgs_train[1])
        '''
        label_truth_train = np.argmax(Y_train, axis=1)
        label_truth_test = np.argmax(Y_test, axis=1)

        save_dir = os.path.join(save_dir, os.path.join(adv_attack)) #, "eps_" + str(eps)))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print (adv_imgs.shape, adv_imgs_test.shape)
        provider.save_h5(adv_imgs, label_truth_train, os.path.join(save_dir, "train_adv.h5"))
        provider.save_h5(adv_imgs_test, label_truth_test, os.path.join(save_dir, "test_adv.h5"))
        # utils.save_h5(X_train, label_truth_train, "FGSM/train_benign.h5")
        # utils.save_h5(X_test, label_truth_test, "FGSM/test_benign.h5")

        '''
        for i in range(adv_labels.shape[0]):
            img = np.squeeze(adv_imgs[i,:,:,:])
            imsave(os.path.join("adv", str(i) + ".jpg"), img)

        for i in range(adv_labels.shape[0]):
            img = np.squeeze(adv_imgs[i,:,:,:])
	    adv_path = "adv_" + str(np.argmax(adv_labels[i,:], axis=0))
	    if not os.path.exists(adv_path):
	        os.makedirs(adv_path)
	    imsave(os.path.join(adv_path, str(i) + ".jpg"), img)
        '''

        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc

    return report


def main(argv=None):
    adv_generate(nb_epochs=FLAGS.nb_epochs, 
                 batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 clean_train=FLAGS.clean_train,
                 nb_filters=FLAGS.nb_filters,
                 data=FLAGS.dataset, 
                 adv_attack=FLAGS.adv_attack,
                 # eps=FLAGS.eps,
                 save_dir=FLAGS.save_dir)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 15, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_string('dataset', 'cifar', 'Evaluation dataset')
    flags.DEFINE_string('adv_attack', 'FGSM', 'Adversarial attacks')
    # flags.DEFINE_float('eps', 0.1, 'maximum distortion of adversarial example')
    flags.DEFINE_string('save_dir', 'data', 'Path of adversarial examples to save')

    tf.app.run()
