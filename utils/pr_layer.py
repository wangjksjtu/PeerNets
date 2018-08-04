from __future__ import absolute_import

from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.layers import Layer, Dropout, LeakyReLU, Lambda, Dense
import tensorflow as tf


def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm

def pairwise_cosine_sim(A_B):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions

    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    A, B = A_B
    A_mag = l2_norm(A, axis=2)
    B_mag = l2_norm(B, axis=2)
    num = K.batch_dot(A, K.permute_dimensions(B, (0,2,1)))
    den = (A_mag * K.permute_dimensions(B_mag, (0,2,1)))
    dist_mat =  num / den

    return dist_mat


class PeerRegularization(Layer):

    def __init__(self, K_, **kwargs):

        self.K_ = K_  # Number of neignbors
        self.supports_masking = False

        super(PeerRegularization, self).__init__(**kwargs)

    def call(self, inputs):
        X = inputs  # Pixel features (N x H x W x F)
        N = tf.shape(X)[0]
        H, W, F = X.get_shape().as_list()[1:]
        # print N, H, W, F
        net = K.reshape(X, [N, H * W, F]) # (N x HW x F)
        # print net
        cos_dists = Lambda(pairwise_cosine_sim)([net, net]) # (N x HW x HW)
        # print cos_dists
        net2 = K.tile(K.expand_dims(net, axis=1), [1, H * W, 1, 1])
        net2 = K.reshape(net2, [N * H * W, H * W, F]) # (NHW x HW x F)
        print net2
        _, idxs = tf.nn.top_k(cos_dists, self.K_) # idxs (N x HW x K)
        idxs = K.reshape(idxs, [N * H * W, self.K_]) # idxs (NHW x K)
        print idxs
        net_neighbors = tf.gather_nd(net2, K.stack([K.transpose(K.tile(K.expand_dims(K.arange(N*H*W), 0), [self.K_, 1])), idxs], 2)) # (NHW x K x F)
        print net_neighbors
        net = K.reshape(X, [N * H * W, F]) # (NHW x F)
        net = tf.tile(K.expand_dims(net, axis=1), [1, self.K_, 1])
        print net
        net = K.reshape(K.stack([net, net_neighbors], axis=2), [N * H * W * self.K_, 2*F])
        net = K.reshape(Dense(F)(net), [N * H * W, self.K_, F]) # (NHW x K x F)
        net = LeakyReLU(alpha=0.2)(K.exp(net))
        net /= K.sum(net, axis=1, keepdims=True) # (NHW x K x F)
        outputs = K.sum(tf.multiply(net, net_neighbors), axis=1)
        outputs = K.reshape(outputs, (N, H, W, F))
        print "-" * 20
        print outputs
        print "-" * 20
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape


class GlobalPeerRegularization(Layer):
    def __init__(self, K_, **kwargs):

        self.K_ = K_  # Number of neignbors
        self.supports_masking = False

        super(GlobalPeerRegularization, self).__init__(**kwargs)

    def call(self, inputs):
        X = inputs  # Pixel features (N x H x W x F)
        N = tf.shape(X)[0]
        H, W, F = X.get_shape().as_list()[1:]
        # print N, H, W, F
        net = K.reshape(X, [N, H * W, F]) # (N x HW x F)
        # print net
        cos_dists = Lambda(pairwise_cosine_sim)([net, net]) # (N x HW x HW)
        # print cos_dists
        net2 = K.tile(K.expand_dims(net, axis=1), [1, H * W, 1, 1])
        net2 = K.reshape(net2, [N * H * W, H * W, F]) # (NHW x HW x F)
        print net2
        _, idxs = tf.nn.top_k(cos_dists, self.K_) # idxs (N x HW x K)
        idxs = K.reshape(idxs, [N * H * W, self.K_]) # idxs (NHW x K)
        print idxs
        net_neighbors = tf.gather_nd(net2, K.stack([K.transpose(K.tile(K.expand_dims(K.arange(N*H*W), 0), [self.K_, 1])), idxs], 2)) # (NHW x K x F)
        print net_neighbors
        net = K.reshape(X, [N * H * W, F]) # (NHW x F)
        net = tf.tile(K.expand_dims(net, axis=1), [1, self.K_, 1])
        print net
        net = K.reshape(K.stack([net, net_neighbors], axis=2), [N * H * W * self.K_, 2*F])
        net = K.reshape(Dense(F)(net), [N * H * W, self.K_, F]) # (NHW x K x F)
        net = LeakyReLU(alpha=0.2)(K.exp(net))
        net /= K.sum(net, axis=1, keepdims=True) # (NHW x K x F)
        outputs = K.sum(tf.multiply(net, net_neighbors), axis=1)
        outputs = K.reshape(outputs, (N, H, W, F))
        print "-" * 20
        print outputs
        print "-" * 20
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape
    

class PeerRegularization_slow(Layer):

    def __init__(self, K_, **kwargs):

        self.K_ = K_  # Number of neignbors
        self.supports_masking = False

        super(PeerRegularization_slow, self).__init__(**kwargs)

    def call(self, inputs):
        X = inputs  # Pixel features (N x H x W x F)
        N = tf.shape(X)[0]
        H, W, F = X.get_shape().as_list()[1:]
        # print N, H, W, F
        net = K.reshape(X, [N, H * W, F]) # (N x HW x F)
        # print net
        cos_dists = Lambda(pairwise_cosine_sim)([net, net]) # (N x HW x HW)
        # print cos_dists
        outputs = []
        for i in range(H * W):
            x_i = net[:, i, :] # (N x F)
            # print x_i
            _, idxs = tf.nn.top_k(cos_dists[:, i, :], self.K_) # idxs (N x K)
            # print idxs
            net_neighbors = tf.gather_nd(net, K.stack([K.transpose(K.tile(K.expand_dims(K.arange(N), 0), [self.K_, 1])), idxs], 2)) # (N x K x F)
            # print net_neighbors

            
            x_o = []
            for j in range(self.K_):
                x_j = net_neighbors[:, j, :] # (N x F)
                # print x_j
                x_merge = K.reshape(K.stack([x_i, x_j], axis=2), [N, 2*F]) # (N x 2F)
                x_ = Dense(F)(x_merge) # (N x F)
                x_ = K.exp(x_)
                x_ = LeakyReLU(alpha=0.2)(x_)
                x_o.append(x_)

            x_o = K.stack(x_o, axis=1) # (N x K x F)
            # print x_o
            x_sum = K.sum(x_o, axis=1, keepdims=True)  # (N x 1 x F)
            # print x_sum
            A = x_o / x_sum # (N x K x F)
            # print A
            outputs.append(K.sum(tf.multiply(A, net_neighbors), axis=1)) # (N x F)
            # print outputs[i]
            

        outputs = K.reshape(K.stack(outputs, axis=1), (N, H, W, F))
        print "-" * 20
        print outputs
        print "-" * 20
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape