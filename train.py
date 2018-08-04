import argparse
import importlib
import os
import sys
import time

import numpy as np
import pandas

# import foolbox
import provider
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
# from tensorflow.python.keras._impl import keras
from cleverhans.attacks import FastGradientMethod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--adv_attack', type=str, default="FGSM", help='Adversarial examples (FGSM/DeepFool/CWL2) [default: FGSM]')
parser.add_argument('--model', type=str, default="all-cnns", help='Model name (all-cnns/vgg/resnet/densenet) [default: all-cnns]')
parser.add_argument('--add_peer', type=bool, default=False, help='Add PeerRegularization [default: False]')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training [default: 128]')
parser.add_argument('--num_epoch', type=int, default=20, help='Batch size during training [default: 100]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.01]')
parser.add_argument('--log_dir', type=str, default="", help="The path of training log (saving directory)")
parser.add_argument('--train_dir', type=str, default="", help="The path of training data (loading directory)")
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')

FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
ADV_ATTACK = FLAGS.adv_attack
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.num_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
ADD_PEER = FLAGS.add_peer
nb_classes = 10

supported_models = ["all-cnns", "resnet", "vgg"] # "densenet bugs!!"
assert (FLAGS.model in supported_models)

TRAIN_DIR = FLAGS.train_dir
LOG_DIR = FLAGS.log_dir
if TRAIN_DIR == "":
    TRAIN_DIR = os.path.join("data", "benign")
if LOG_DIR == "":
    LOG_DIR = os.path.join("logs", os.path.join(ADV_ATTACK, FLAGS.model))
if ADD_PEER:
    LOG_DIR = os.path.join(LOG_DIR, "PR_" + str(time.time()))
else:
    LOG_DIR = os.path.join(LOG_DIR, "NPR_" + str(time.time()))

TEST_DIR = os.path.join("data", ADV_ATTACK)

print ("train_dir: " + TRAIN_DIR)
print ("test_dir: " + TEST_DIR)
print ("log_dir: " + LOG_DIR)

assert (os.path.exists(TRAIN_DIR))
# assert (os.path.exists(TEST_DIR))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join('models', FLAGS.model + '.py')
os.system('cp %s %s' % ("train.py", LOG_DIR)) # bkp of train procedure
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def

FLAGS.train_dir = TRAIN_DIR
FLAGS.log_dir = LOG_DIR
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


def get_loss(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int64, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss


def save_summary(model, header, suffix):
     assert(suffix.split(".")[0] == "")
     with open(header + suffix, 'w') as fh:
         # Pass the file handle in as a lambda functions to make it callable
         model.summary(print_fn=lambda x: fh.write(x + '\n'))


class EarlyStopping(Callback):
    def __init__(self, monitor='acc', value=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        # print (current)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor)
            exit()

        if current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def load_cifar10():
    print "[Loading data ...]"
    X_train, Y_train = provider.load_data(TRAIN_DIR, "train_benign.h5")
    X_train, Y_train, _ = provider.shuffle_data(X_train, Y_train)

    X_val, Y_val = provider.load_data(TRAIN_DIR, "test_benign.h5")
    X_val, Y_val, _ = provider.shuffle_data(X_val, Y_val)

    X_test, Y_test = provider.load_data(TEST_DIR, "test_adv.h5")
    X_test, Y_test, _ = provider.shuffle_data(X_test, Y_test)
    print (X_train.shape, Y_train.shape)
    print (X_val.shape, Y_val.shape)
    # print (X_test.shape, Y_test.shape)
    print "[Finish Loading]"
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def train():
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = load_cifar10()
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_val = np_utils.to_categorical(Y_val, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    print (X_train.shape, Y_train.shape)
    model = MODEL.get_model(nb_classes, ADD_PEER)
    if OPTIMIZER == "adam":
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    else:
        sgd = SGD(lr=BASE_LEARNING_RATE, decay=DECAY_RATE, momentum=MOMENTUM, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    save_summary(model, os.path.join(LOG_DIR, "model_arch"), ".txt")
    # plot_model(model, to_file="parameters/model" + ".pdf", show_shapes=True)
    callbacks = [
        # EarlyStopping(monitor='acc', value=0.998, verbose=1),
        ModelCheckpoint(filepath=os.path.join(LOG_DIR, "keras_weights.hdf5"),
                        monitor='acc', verbose=0,
                        save_weights_only=False, mode='max')
    ]

    print (X_val.shape, Y_val.shape)
    history_callback = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCH, shuffle=True,
                                verbose=2, callbacks=callbacks, validation_data=(X_test, Y_test))  # validation_data=(X_val, Y_val),
    pandas.DataFrame(history_callback.history).to_csv(os.path.join(LOG_DIR, "keras_history.csv"))
    print (max(history_callback.history['val_acc']))

    '''
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

    fgsm = FastGradientMethod(model, sess=sess)
    fgsm_params = {'eps': 0.01, 'clip_min': 0., 'clip_max': 1.}
    adv_x = fgsm.generate(x, **fgsm_params)
    # preds_adv = model(adv_x)
    print (adv_x.shape) # preds_adv.shape

    X_val_adv = []
    for i in range(10):
        adv_imgs_tmp = sess.run(adv_x, feed_dict={x: X_val[i*1000: (i+1)*1000]})
        X_val_adv.append(adv_imgs_tmp)
    X_val_adv = np.vstack(X_val_adv)

    '''
    '''
    print X_val.shape, Y_val.shape
    fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))
    attack = foolbox.attacks.FGSM(fmodel)

    adversarial_images=[]
    for i in range(X_val.shape[0]):
        # print X_val[i].shape, Y_val[i].shape
        adversarial = attack(X_val[i], np.argmax(Y_val[i]))
        adversarial_images.append(adversarial)
        if i % 100 == 0:
            print ("100 finished!")

    adversarial_images = np.stack(adversarial_images, axis=0)
    print adversarial_images.shape
    '''
    Y_pred = model.predict(X_test, verbose=0)
    print np.mean(np.argmax(Y_pred, axis=1) ==  np.argmax(Y_test, axis=1))
    # model.save(os.path.join(log_dir, 'model.h5'))


def input_function(features,labels=None,shuffle=False):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"img_input": features},
        y=labels,
        shuffle=shuffle
    )
    return input_fn


def train_estimator():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = load_cifar10()
            Y_train = np_utils.to_categorical(Y_train, nb_classes)
            Y_val = np_utils.to_categorical(Y_val, nb_classes)
            Y_test = np_utils.to_categorical(Y_test, nb_classes)
            Y_train = Y_train.astype(np.float32)
            Y_val = Y_val.astype(np.float32)
            Y_test = Y_test.astype(np.float32)
            print Y_train.shape, Y_val.shape
            y_ = tf.placeholder(tf.int64, [BATCH_SIZE])
            x = tf.placeholder(tf.float32, [BATCH_SIZE, 32, 32, 3])

            batch = tf.Variable(0, name='global_step', trainable=False)

            # build the network
            model = MODEL.get_model(nb_classes)
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            estimator_model = tf.keras.estimator.model_to_estimator(keras_model=model)

            for epoch in np.arange(MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                estimator_model.train(input_fn=input_function(X_train, Y_train, True))
                score = estimator_model.evaluate(input_function(X_val, labels=Y_val))
                print score

def train_tf():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = load_cifar10()
            y_ = tf.placeholder(tf.int64, [BATCH_SIZE])
            x = tf.placeholder(tf.float32, [BATCH_SIZE, 32, 32, 3])

            batch = tf.Variable(0, name='global_step', trainable=False)

            # build the network
            logits = MODEL.get_model(nb_classes)(x)

            '''
            logits = MODEL.get_model(x)
            print logits, y_
            '''

            batch = tf.Variable(0, name='global_step', trainable=False)

            loss = get_loss(logits, y_)
            tf.summary.scalar('loss', loss)
            print loss

            correct = tf.equal(tf.argmax(logits, 1), y_)
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            with tf.Session(config=config) as sess:
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess= sess, coord=coord)

                summary_op = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
                val_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'val'), sess.graph)
                test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

                ops = {'images_pl': x,
                       'labels_pl': y_,
                       'logits': logits,
                       'loss': loss,
                       'train_op': train_op,
                       'merged': summary_op,
                       'step': batch}

                try:
                    acc_eval_max = 0
                    acc_test_max = 0

                    for epoch in np.arange(MAX_EPOCH):
                        if coord.should_stop():
                            break

                        log_string('**** EPOCH %03d ****' % (epoch))
                        sys.stdout.flush()

                        train_one_epoch(sess, ops, train_writer, X_train, Y_train)
                        acc_eval = eval_one_epoch(sess, ops, val_writer, X_val, Y_val)
                        acc_test = eval_one_epoch(sess, ops, test_writer, X_test, Y_test, "test")

                        if acc_eval_max < acc_eval:
                            save_path = saver.save(sess, os.path.join(LOG_DIR, "model_eval.ckpt"))
                            acc_eval_max = acc_eval
                        if acc_test_max < acc_test:
                            save_path = saver.save(sess, os.path.join(LOG_DIR, "model_test.ckpt"))
                            acc_test_max = acc_test

                        # Save the variables to disk.
                        if epoch % 10 == 0:
                            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                            log_string("Model saved in file: %s" % save_path)

                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                finally:
                    coord.request_stop()
                coord.join(threads)


def train_one_epoch(sess, ops, train_writer, current_data, current_label):
    """ ops: dict mapping from string to tf ops """
    train_size = current_label.shape[0]
    num_batches = train_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['images_pl']: current_data[start_idx:end_idx,...],
                        ops['labels_pl']: current_label[start_idx:end_idx]}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['logits']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        # print pred_val, current_label[start_idx:end_idx]
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val

    # print (loss_sum, num_batches)
    # print (total_correct, total_seen)
    log_string('Train')
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
    return total_correct / float(total_seen)


def eval_one_epoch(sess, ops, val_writer, current_data, current_label, des="eval"):
    """ ops: dict mapping from string to tf ops """
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(nb_classes)]
    total_correct_class = [0 for _ in range(nb_classes)]

    val_size = current_label.shape[0]
    num_batches = val_size // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['images_pl']: current_data[start_idx:end_idx, ...],
                        ops['labels_pl']: current_label[start_idx:end_idx]}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['logits']], feed_dict=feed_dict)
        val_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val * BATCH_SIZE)
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)

    log_string('-' * 20)
    if des == "eval":
        log_string('Eval:')
        log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
        log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
        log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class) / np.array(total_seen_class,dtype=np.float))))
    else:
        log_string('Test: ' + ADV_ATTACK)
        log_string('test (adv) mean loss: %f' % (loss_sum / float(total_seen)))
        log_string('test (adv) accuracy: %f'% (total_correct / float(total_seen)))
        log_string('test (adv) avg class acc: %f' % (np.mean(np.array(total_correct_class) / np.array(total_seen_class,dtype=np.float))))

    return total_correct / float(total_seen)

if __name__ == "__main__":
    train()
