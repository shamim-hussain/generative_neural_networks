import tensorflow as tf
from keras.layers import *
from keras.metrics import binary_crossentropy, mean_squared_error
from keras import backend as K
from functools import reduce


class MaxPool2DWithArgmax(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='SAME', **kwargs):
        self._kernel = (1, *pool_size, 1)
        if not strides:
            strides = pool_size
        self._stride = (1, *strides, 1)
        self._padding = padding
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        super().build(input_shape, **kwargs)
        self.built = True

    def call(self, input):
        [pooled, argmax] = tf.nn.max_pool_with_argmax(input, self._stride, self._kernel, self._padding)
        self._argmax = argmax
        return pooled

    def compute_output_shape(self, input_shape):
        xs, ys = self._stride[1:3]
        return (input_shape[0], (input_shape[1] + xs - 1) // xs, (input_shape[2] + ys - 1) // ys, input_shape[3])


class MaxUnpool2DWithArgmax(Layer):
    def __init__(self, pool_layer: MaxPool2DWithArgmax, **kwargs):
        self._pool_layer = pool_layer
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        super().build(input_shape, **kwargs)
        self.built = True

    def call(self, input):
        pf = tf.reshape(input, (-1,))

        shp = tf.shape(self._pool_layer.input)
        shp = tf.cast(shp, tf.int64)
        shpf = tf.reduce_prod(shp)

        b_argmax = self._pool_layer._argmax + \
                   (tf.range(shp[0], dtype=tf.int64) * tf.reduce_prod(shp[1:])) \
                       [:, tf.newaxis, tf.newaxis, tf.newaxis]
        amf = tf.reshape(b_argmax, (-1, 1))

        upf = tf.scatter_nd(amf, pf, (shpf,))

        unpooled = tf.reshape(upf, shp)
        return unpooled

    def compute_output_shape(self, input_shape):
        return self._pool_layer.input_shape


class MSELossLayer(Layer):
    def __init__(self, lmbda=1, **kwargs):
        self.is_placeholder = True
        self.lmbda = K.variable(lmbda)
        super().__init__(**kwargs)

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        self.add_loss(self.lmbda * K.mean(mean_squared_error(y, x)), inputs=inputs)
        # We won't actually use the output.
        return x


class XEntropyLossLayer(Layer):
    def __init__(self, lmbda=1, **kwargs):
        self.is_placeholder = True
        self.lmbda = K.variable(lmbda)
        super().__init__(**kwargs)

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        self.add_loss(self.lmbda * K.mean(binary_crossentropy(y, x)), inputs=inputs)
        # We won't actually use the output.
        return x


def seq_call(input: tf.Tensor, layers: list):
    return reduce(lambda a, x: x(a), layers, input)


def conv_block(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               dropout=.2, bn=True, dilation_rate=(1, 1), name='conv_block', padding='same', **kwargs):
    if not hasattr(conv_block, 'counter'):
        conv_block.counter = 0
    layers = []
    layers.append(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                         padding=padding, activation=None, use_bias=not bn, dilation_rate=dilation_rate,
                         name=name + '_%d_conv' % conv_block.counter, **kwargs))
    if bn:
        layers.append(BatchNormalization(name=name + '_%d_batchnorm' % conv_block.counter))
    if activation:
        layers.append(Activation(activation=activation, name=name + '_actvn_%d' % conv_block.counter))
    if dropout:
        layers.append(Dropout(.2, name=name + '_%d_drpout' % conv_block.counter))

    conv_block.counter += 1
    return layers


def deconv_block(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                 dropout=0., bn=True, dilation_rate=(1, 1), name='deconv_block',padding='same', **kwargs):
    if not hasattr(conv_block, 'counter'):
        conv_block.counter = 0
    layers = []
    layers.append(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                  padding=padding, activation=None, use_bias=not bn, dilation_rate=dilation_rate,
                                  name=name + '_%d_conv' % conv_block.counter, **kwargs))
    if bn:
        layers.append(BatchNormalization(name=name + '_%d_batchnorm' % conv_block.counter))
    if activation:
        layers.append(Activation(activation=activation, name=name + '_actvn_%d' % conv_block.counter))
    if dropout:
        layers.append(Dropout(.2, name=name + '_%d_drpout' % conv_block.counter))

    conv_block.counter += 1
    return layers


def dense_block(nodes: list, activations: list, dropouts: list, name='dense_block'):
    if not hasattr(dense_block, 'counter'):
        dense_block.counter = 0
    if len(activations) == 1:
        activations *= len(nodes)
    if len(dropouts) == 1:
        dropouts *= len(nodes)

    layers = []
    for node, activation, dropout, i in zip(nodes, activations, dropouts, range(len(nodes))):
        if dropout:
            layers.append(Dropout(dropout, name=name + '_%d_drpout_%d' % (dense_block.counter, i)))
        layers.append(Dense(node, activation=activation, name=name + '_%d_dense_%d' % (dense_block.counter, i)))
    dense_block.counter += 1
    return layers


class VAESampling(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def call(self, inputs):
        z_mean0, z_log_sigma0 = inputs
        z_sigma0 = tf.exp(z_log_sigma0)
        input_shape = tf.shape(z_mean0)
        flat_shape = [input_shape[0], -1]

        z_mean = tf.reshape(z_mean0, flat_shape)
        z_log_sigma = tf.reshape(z_log_sigma0, flat_shape)
        z_sigma = tf.reshape(z_sigma0, flat_shape)

        kl_losses = 0.5 * tf.reduce_sum(tf.square(z_mean) + z_sigma - z_log_sigma - 1, axis=1)
        kl_loss = tf.reduce_mean(kl_losses)
        self.add_loss(kl_loss)

        z0 = tf.random_normal(input_shape)
        return z_mean0 + z_sigma0 * z0


class VAERecLossLayer(Layer):
    def __init__(self, lmbda=1., **kwargs):
        self.is_placeholder = True
        self.lmbda = K.variable(lmbda)
        super().__init__(**kwargs)

    def call(self, inputs):
        y, x = inputs
        shp = tf.shape(x)
        fshp = [shp[0], -1]
        x = tf.reshape(x, fshp)
        y = tf.reshape(y, fshp)

        y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)
        rec_losses = tf.reduce_sum(-x * tf.log(y) - (1 - x) * tf.log(1 - y), 1)
        rec_loss = tf.reduce_mean(rec_losses)
        self.add_loss(self.lmbda * rec_loss)
        return y
