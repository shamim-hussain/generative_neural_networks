import tensorflow as tf
from keras.layers import *
from keras.metrics import binary_crossentropy, mean_squared_error
from keras import backend as K
from functools import reduce
from keras.initializers import constant
from keras.models import *

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
    def __init__(self, lmbda=1., **kwargs):
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
    def __init__(self, lmbda=1., **kwargs):
        self.is_placeholder = True
        self.lmbda = K.variable(lmbda)
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
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


def sep_conv_block(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                   dropout=.2, bn=True, dilation_rate=(1, 1), name='sep_conv_block', padding='same', **kwargs):
    if not hasattr(sep_conv_block, 'counter'):
        sep_conv_block.counter = 0
    layers = []
    layers.append(SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                  padding=padding, activation=None, use_bias=not bn, dilation_rate=dilation_rate,
                                  name=name + '_%d_conv' % sep_conv_block.counter, **kwargs))
    if bn:
        layers.append(BatchNormalization(name=name + '_%d_batchnorm' % sep_conv_block.counter))
    if activation:
        layers.append(Activation(activation=activation, name=name + '_actvn_%d' % sep_conv_block.counter))
    if dropout:
        layers.append(Dropout(.2, name=name + '_%d_drpout' % sep_conv_block.counter))

    sep_conv_block.counter += 1
    return layers

def deconv_block(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                 dropout=0., bn=True, dilation_rate=(1, 1), name='deconv_block',padding='same', **kwargs):
    if not hasattr(deconv_block, 'counter'):
        deconv_block.counter = 0
    layers = []
    layers.append(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                  padding=padding, activation=None, use_bias=not bn, dilation_rate=dilation_rate,
                                  name=name + '_%d_conv' % deconv_block.counter, **kwargs))
    if bn:
        layers.append(BatchNormalization(name=name + '_%d_batchnorm' % deconv_block.counter))
    if activation:
        layers.append(Activation(activation=activation, name=name + '_actvn_%d' % deconv_block.counter))
    if dropout:
        layers.append(Dropout(.2, name=name + '_%d_drpout' % deconv_block.counter))

    deconv_block.counter += 1
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
        y0, x0 = inputs
        shp = tf.shape(x0)
        fshp = [shp[0], -1]
        x = tf.reshape(x0, fshp)
        y = tf.reshape(y0, fshp)

        y = tf.clip_by_value(y, 1e-7, 1 - 1e-7)
        rec_losses = tf.reduce_sum(-x * tf.log(y) - (1 - x) * tf.log(1 - y), 1)
        rec_loss = tf.reduce_mean(rec_losses)
        self.add_loss(rec_loss)
        return y0


class MovingAvg(Layer):
    def __init__(self, mu_init=0., sigma_init=1., momentum=0.99, eps=1e-3, **kwargs):
        self._mu_init = mu_init
        self._sigma_init = sigma_init
        self._mu_l = momentum
        self._sigma_l = momentum
        self._eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.mu = self.add_weight('mu', input_shape[1:], dtype=tf.float32,
                                  initializer=constant(self._mu_init),
                                  trainable=False)
        self.sigma = self.add_weight('sigma', input_shape[1:], dtype=tf.float32,
                                     initializer=constant(self._sigma_init),
                                     trainable=False)
        super().build(input_shape)
        self.built = True

    def call(self, x):
        mean, var = tf.nn.moments(x, [0])
        self.add_update([K.moving_average_update(self.mu, mean, self._mu_l),
                         K.moving_average_update(self.sigma, tf.sqrt(var), self._sigma_l)], x)
        return (x-self.mu) / (self.sigma+self._eps)

def depth_block(activation='relu', dropout=.2, bn=True, name='depth_block'):
    if not hasattr(depth_block, 'counter'):
        depth_block.counter = 0
    layers=[]
    if bn:
        layers.append(BatchNormalization(name=name + '_%d_batchnorm' % depth_block.counter))
    if activation:
        layers.append(Activation(activation=activation, name=name + '_actvn_%d' % depth_block.counter))
    if dropout:
        layers.append(Dropout(.2, name=name + '_%d_drpout' % depth_block.counter))

    depth_block.counter += 1
    return layers



class CausalConv1D(Conv1D):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        ks = self.kernel_size[0]
        kernel_shape = ((ks+1)//2, input_dim, self.filters)
        zeros_shape = (ks//2, input_dim, self.filters)

        kernel_var = self.add_weight(shape=kernel_shape,
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        kernel_cons = K.zeros(zeros_shape, dtype=K.floatx())
        self.kernel = K.concatenate([kernel_var,kernel_cons], axis=0)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        

class SigTanActivation(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        self.built = True

    def call(self, x: tf.Tensor):
        if x.shape.as_list()[-1]%2:
            raise ValueError('Number of features must be even!given='+
                             str(x.shape.as_list()[-1]))
        x1, x2 = tf.split(x, 2, axis=-1)
        d = tf.nn.tanh(x1)
        m = tf.nn.sigmoid(x2)
        return d*m

    def compute_output_shape(self, input_shape):
        if input_shape[-1]%2:
            raise ValueError('Number of features must be even! given='+
                             str(x.shape.as_list()[-1]))
        return (*input_shape[:-1], int(input_shape[-1] / 2))


class AdvLossLayer(Layer):
    def __init__(self, lmbda=1., **kwargs):
        self.is_placeholder = True
        self.lmbda = K.variable(lmbda)
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs
        y = tf.ones(tf.shape(x))
        aloss = self.lmbda * K.mean(binary_crossentropy(y, x))
        aacc = K.mean(K.equal(y, K.round(x)), axis=-1)
        self.add_loss(aloss, inputs=inputs)
        # We won't actually use the output.
        return [aloss, aacc]
    
    def compute_output_shape(self, input_shape):
        return [(),()]


class DiscLossLayer(Layer):
    def __init__(self, lmbda=1., **kwargs):
        ''' (reals,fakes) '''
        self.is_placeholder = True
        self.lmbda = K.variable(lmbda)
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list) or isinstance(inputs, tuple)
        assert len(inputs) == 2
        reals, fakes = inputs
        x = tf.concat((reals,fakes), axis=0)
        y =  tf.concat((tf.ones_like(reals), tf.zeros_like(fakes)), axis=0)
        dloss = self.lmbda * K.mean(binary_crossentropy(y, x))
        dacc = K.mean(K.equal(y, K.round(x)), axis=-1)
        self.add_loss(dloss, inputs=inputs)
        # We won't actually use the output.
        return [dloss, dacc]
    
    def compute_output_shape(self,input_shape):
        return [(),()]


class MAELossLayer(Layer):
    def __init__(self, lmbda=1., **kwargs):
        self.is_placeholder = True
        self.lmbda = K.variable(lmbda)
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        y = inputs[1]
        self.add_loss(self.lmbda * K.mean(K.abs(y - x)), inputs=inputs)
        # We won't actually use the output.
        return x
    

def build_model_on(in_t, base_model, copy=False, set_weights=False, name=None):
    x = in_t
    tdict = {base_model.get_input_at(0): x}
    lrs = list(lr for lr in base_model.layers if not isinstance(lr, InputLayer))
    for lr in lrs:
        l_in = lr.get_input_at(0)
        l_out = lr.get_output_at(0)
        
        in_t = [tdict[t] for t in l_in]\
                        if isinstance(l_in, list)\
                        else tdict[l_in]
        if copy:
            cg = lr.get_config()
            if name:
                cg['name'] = name+'_'+cg['name']
            else:
                del cg['name']
            mlr = lr.__class__.from_config(cg)
            if set_weights:
                mlr.set_weights(lr.get_weights())
            lr = mlr
        
        out_t = lr(in_t)
        tdict[l_out] = out_t
    return tdict[base_model.get_output_at(0)]


def copy_model(base_model, set_weights=False, name=None):
    iname = name+'_Input' if name else None
    in_t = Input(batch_shape=base_model.get_input_shape_at(0), name = iname)
    out_t = build_model_on(in_t, base_model, set_weights=set_weights, name=name)
    return Model(in_t, out_t)
 
    

    