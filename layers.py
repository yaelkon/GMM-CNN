import math
import tensorflow as tf
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
from keras import backend as K
from keras.models import Model
from keras.layers import Layer
from keras.initializers import constant, random_normal
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton
from keras.constraints import  Constraint

seed(101)
set_random_seed(101)


class MinValue(Constraint):
    """Constrains the weights to be grater than min.
    """
    def __init__(self, min_value=0.1):
        self.min = min_value

    def __call__(self, w):
        return K.cast(K.maximum(w, self.min), K.floatx())


class GMM(Layer):
    def __init__(self, n_clusters=2, epsilon=0.1, seed=101, modeled_layer_type='Conv2d', **kwargs):
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.seed = seed
        self.mu = None
        self.std = None
        self.alpha = None
        self.modeled_layer_type = modeled_layer_type

        super(GMM, self).__init__(**kwargs)

    def build(self, input_shape, x=None):
        """       Create a trainable weight variable for this layer.
            x must be a tensor object
            input_shape - must have the shape (batch, height, width, channels) according to "channel_last" of Conv2D layer
            reshape_input_shape - BHW * D

            Note that the sigma values are saves as std, not as variance"""

        self.mu = self.add_weight( name='mu',
                                   shape=(self.n_clusters, input_shape[-1]),
                                   initializer=random_normal(mean=0, stddev=0.4, seed=self.seed),
                                   trainable=True )

        self.std = self.add_weight( name='std',
                                    shape=(self.n_clusters, input_shape[-1]),
                                    initializer=random_normal(mean=0.3, stddev=0.05, seed=self.seed),
                                    trainable=True,
                                    constraint=MinValue(min_value=self.epsilon) )

        self.alpha = self.add_weight( name='alpha',
                                      shape=(self.n_clusters,),
                                      initializer=constant( value=(1 / self.n_clusters) ),
                                      trainable=True )
        super(GMM, self).build(input_shape)

    def call(self, x, **kwargs):
        """     x must be a tensor object
            input_shape - must have the shape (batch, height, width, channels) according to "channel_last" of Conv2D layer
            reshape_input_shape - BHW * D
        """
        # Preventing from a gradient explosion
        self.std = K.maximum(self.std, self.epsilon)
        self.alpha = K.exp(self.alpha)
        self.alpha = self.alpha / K.sum(self.alpha)

        x_dim = K.shape(x)

        if self.modeled_layer_type == 'Conv2d':
            pool_dim = K.stack([x_dim[0], x_dim[1], x_dim[2], x_dim[3]])
            B_dim = pool_dim[0]
            H_dim = pool_dim[1]
            W_dim = pool_dim[2]
            D_dim = pool_dim[3]
            n_samples = B_dim * H_dim * W_dim

        elif self.modeled_layer_type == 'Dense':
            pool_dim = K.stack([x_dim[0], x_dim[1]])
            B_dim = pool_dim[0]
            D_dim = pool_dim[1]
            n_samples = B_dim

        # Preparing matrices
        x = K.reshape(x, (n_samples, D_dim))
        x_rep = K.repeat_elements( x, self.n_clusters, axis=0 )
        mu_rep = K.tile(self.mu, (n_samples,1))
        sigma_rep = K.tile(self.std, (n_samples,1))
        alpha_rep = K.tile(self.alpha, (n_samples,))

        # Calculating the log likelihood function in log scale
        dist = K.square(x_rep - mu_rep) / K.square(sigma_rep)
        exponent = -(1 / 2.0) * dist
        exponent = K.maximum(exponent, -10)
        cons = 2 * math.pi
        exp_coeff = (1 / (K.sqrt(cons * K.square(sigma_rep))))
        prob = exp_coeff * K.exp(exponent)

        l_p_k = K.sum(K.log(prob), axis=1)
        l_a_k = K.log(alpha_rep)
        s_k = l_a_k + l_p_k

        if self.modeled_layer_type == 'Conv2d':
            s_k = K.reshape( s_k, (B_dim, H_dim, W_dim, self.n_clusters) )
        elif self.modeled_layer_type == 'Dense':
            s_k = K.reshape( s_k, (B_dim, self.n_clusters) )

        return s_k

    def compute_output_shape(self, input_shape):
        # output_shape = (1,)
        if self.modeled_layer_type == 'Conv2d':
            output_shape = (input_shape[0], input_shape[1], input_shape[2], self.n_clusters)
        elif self.modeled_layer_type == 'Dense':
            output_shape = (input_shape[0], self.n_clusters)

        return output_shape

    def get_config(self):
        config = {
            'n_clusters': self.n_clusters,
            'modeled_layer_type': self.modeled_layer_type,
            'epsilon': self.epsilon
        }
        base_config = super(GMM, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def get_layer_output(keras_model, layer_name, input_data, batch_size=32):
    """obtain the output of an intermediate layer"""

    intermediate_layer_model = Model(inputs=keras_model.get_input_at(0),
                                     outputs=keras_model.get_layer(name=layer_name).get_output_at(0))
    intermediate_output = intermediate_layer_model.predict(input_data, batch_size=batch_size)

    return intermediate_output


def _collect_input_shape(input_tensors):
    """Collects the output shape(s) of a list of Keras tensors.

    # Arguments
        input_tensors: list of input tensors (or single input tensor).

    # Returns
        List of shape tuples (or single tuple), one tuple per input.
    """
    input_tensors = to_list(input_tensors)
    shapes = []
    for x in input_tensors:
        try:
            shapes.append(K.int_shape(x))
        except TypeError:
            shapes.append(None)
    return unpack_singleton(shapes)


def gmm_bayes_activation(TLL):
    """Computes the probability P(h=k|x), where TLL is the output of gmm layer is given by log(P(x,h=k))"""

    K_dim = TLL.get_shape()[-1]
    max_TLL = K.max(TLL, axis=-1)
    max_TLL = K.repeat_elements(K.expand_dims(max_TLL, axis=-1), K_dim, axis=-1)
    ETLL = K.exp(TLL - max_TLL)
    SETLL = K.sum(ETLL, -1)
    rep_SETLL = K.repeat_elements(K.expand_dims(SETLL, axis=-1), K_dim, axis=-1)

    depended_prob = ETLL / rep_SETLL

    return depended_prob

def max_channel_histogram(inputs):

    # argmax_channel = K.argmax(inputs, axis=-1)
    # argmax_channel = K.cast(argmax_channel, dtype='int32')
    # zeros = K.zeros_like(inputs)
    # m, n = inputs.get_shape()[1:3]
    # X = K.expand_dims(K.expand_dims(K.arange(batch_size), axis=-1), axis=-1)
    # # X = K.expand_dims(K.arange(batch_size), axis=[1,2])
    # Y = K.expand_dims(K.expand_dims(K.arange(m), axis=0), axis=-1)
    # Z = K.expand_dims(K.expand_dims(K.arange(n), axis=0), axis=0)
    # zeros[X, Y, Z, argmax_channel] = 1
    # channel_hist = K.sum(zeros, axis=(1, 2))

    one_zero_tens = tf.where(tf.equal(tf.reduce_max(inputs, axis=-1, keep_dims=True), inputs), tf.constant(1, shape=inputs.shape),
                    tf.constant(0, shape=inputs.shape))
    one_zero_tens = K.cast(one_zero_tens, dtype='float32')

    return one_zero_tens