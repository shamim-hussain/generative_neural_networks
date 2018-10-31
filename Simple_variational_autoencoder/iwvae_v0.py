# -*- coding: utf-8 -*-


from keras.datasets import mnist
from keras.layers import *
from keras.optimizers import RMSprop, Adam
from keras.models import *
from keras import backend as K
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pytictoc import TicToc
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback

K.clear_session()

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

ldim = 3
bsize=256
num_epochs = 200

in_t = Input(X_train.shape[1:])
x = in_t
x = Flatten()(x)
f_in = x
x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)
mu = Dense(ldim)(x)
ls = Dense(ldim)(x)

enc_model = Model(in_t, mu)


class Sampling_layer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True
    def call(self, inputs, **kwargs):
        mu, ls = inputs
        eps = tf.random_normal(tf.shape(mu))
        sigma = tf.exp(ls)
        z = eps*sigma + mu
        lpost = -tf.reduce_sum(ls+.5*eps**2, axis=-1)
        lprior = -tf.reduce_sum(.5*z**2, axis=-1)
        floss = tf.reduce_mean(lpost-lprior)*5
        self.add_loss(floss, inputs)
        return z
    def compute_output_shape(self, input_shape):
        return input_shape[0]


z_samp = Sampling_layer()([mu, ls])
z_in = Input((ldim,))
x = z_in
x = Dense(128, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(28*28, activation='sigmoid')(x)
x = Reshape(X_train.shape[1:])(x)
d_out = x
dec_model = Model(z_in, d_out)

class XELoss_layer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        super().build(input_shape)
        self.built = True
    def call(self, inputs, **kwargs):
        o, t = inputs
        o = tf.reshape(o, [tf.shape(o)[0],-1])
        t = tf.reshape(t, [tf.shape(t)[0],-1])
        XEloss = K.mean(K.sum(K.binary_crossentropy(t, o), axis=-1))
        self.add_loss(XEloss)
        return o

x = dec_model(z_samp)
x = XELoss_layer()([x, f_in])
out_t = x


model = Model(in_t, out_t)
print(model.summary())

opt = Adam(1e-5)
model.compile(opt, None, None)


from scipy.stats import norm
def plot_samples(epoch, logs):
    if epoch % 10:
        return
    plt.figure(figsize=(10,10))
    num_r = 5
    gp = norm.ppf(np.linspace(.05,.95, num_r))
    mg = np.stack(np.meshgrid(*[gp]*ldim), axis=-1)
    fakes = dec_model.predict_on_batch(mg.reshape(-1,ldim))
    ipr = int(np.sqrt(num_r**ldim))
    myimg = np.zeros([ipr*28]*2)
    for k in range(ipr*ipr):
        i = k//ipr
        j = k%ipr
        myimg[i*28:(i*28+28),j*28:(j*28+28)]=fakes[k]
    plt.imshow(myimg, 'gray')
    plt.pause(.01)
    figs=plt.get_fignums()
    if len(figs)>5: 
        for ff in figs[0:len(figs)-5]: plt.close(ff)
            

ps_cb = LambdaCallback(on_epoch_end=plot_samples)

model.fit(X_train, None, batch_size=bsize, epochs=num_epochs, 
          validation_data=[X_test, None], callbacks=[ps_cb])





