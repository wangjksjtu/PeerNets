import h5py
from PIL import Image
import os
from keras.datasets import mnist
import argparse
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="FGSM", help='Path of input folder [default:FGSM]')

FLAGS = parser.parse_args()
INPUT_DIR = FLAGS.input
assert (os.path.exists(INPUT_DIR))

num_counts = [0] * 10

def restore_images(input_dir, out_dirs, quality = 100):
    if input_dir == "benign": mnist = get_mnist_adv(input_dir, False)
    else: mnist = get_mnist_adv(input_dir)
    # print images_train.shape, labels_train.shape
    # print images_test.shape, labels_test.shape
    for index, save_dir in enumerate(out_dirs):
        print save_dir
        for i in range(10):
            if not os.path.exists(os.path.join(save_dir, str(i))):
                os.makedirs(os.path.join(save_dir, str(i)))

        counts = [0] * 10

        images, labels = mnist[index]
        # print labels
        print (images.shape, labels.shape)
        # print np.sum(labels == 2)
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i, ...])
            label = labels[i]
            num_counts[label] += 1
            counts[label] += 1
            outname = os.path.join(str(label), str(counts[label]) + ".jpeg")
            # if img.mode != 'RGB':
            #     img = img.convert('RGB')
            img.save(os.path.join(save_dir, outname), quality=quality)

            # if i == 100:
            #     break

def save_h5(data, label, filename):
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=data, **comp_kwargs)
        f.create_dataset('label', data=label, **comp_kwargs)


def read_h5(filename):
    f = h5py.File(filename,'r')
    data, label = f['data'], f['label']
    return data.value, label.value


def get_mnist_adv(input_dir='FGSM', adv=True):
    # print np.sum(Y_train == 1) + np.sum(Y_test == 1)
    if adv:
        X_train, Y_train = read_h5(os.path.join(input_dir, "train_adv.h5"))
        X_test, Y_test = read_h5(os.path.join(input_dir, "test_adv.h5"))
    else:
        X_train, Y_train = read_h5(os.path.join(input_dir, "train_benign.h5"))
        X_test, Y_test = read_h5(os.path.join(input_dir, "test_benign.h5"))

    X_train, X_test = np.squeeze(X_train), np.squeeze(X_test)
    X_train *= 255
    X_test  *= 255
    X_train = X_train.astype('uint8')
    X_test = X_test.astype('uint8')

    # print X_train.shape, Y_train.shape
    # print X_test.shape, Y_test.shape
    return (X_train, Y_train), (X_test, Y_test)


if __name__ == "__main__":
    train_dir = os.path.join(INPUT_DIR, "train")
    test_dir = os.path.join(INPUT_DIR, "test")
    OUTPUT_DIRS = [train_dir, test_dir]
    restore_images(INPUT_DIR, OUTPUT_DIRS)
    print num_counts
