import numpy as np
import os
import h5py

def load_data(img_dirs, h5_filename="data.h5"):
    f = os.path.join(img_dirs, h5_filename)
    data, label = read_h5(f)
    return data.value, label.value

def save_h5(data, label, filename):
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=data, **comp_kwargs)
        f.create_dataset('label', data=label, **comp_kwargs)


def read_h5(filename):
    f = h5py.File(filename,'r')
    data, label = f['data'], f['label']
    return data, label


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
            data: B,... numpy array
            label: B, numpy array
        Return:
            shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


if __name__ == "__main__":
    f = os.path.join("data/quality_100", "train.h5")
    data, label = read_h5(f)
    print (data.value.shape, label.value.shape)