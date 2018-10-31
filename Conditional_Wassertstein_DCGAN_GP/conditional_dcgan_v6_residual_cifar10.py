from keras.datasets import cifar10
from keras.layers import *
from keras.optimizers import RMSprop, Adam
from keras.models import *
from keras import backend as K
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pytictoc import TicToc
from keras.utils import to_categorical

K.clear_session()

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

bsize = 32


def gen_samp(bsize):
    lx = len(X_train)
    ind = np.arange(lx)
    while True:
        np.random.shuffle(ind)
        for k in range(0, lx, bsize):
            ss = np.arange(k, min(k + bsize, lx))
            if k + bsize > lx:
                ss = np.append(ss, np.arange(k + bsize - lx))
            yield X_train[ind[ss]], Y_train[ind[ss]]


sdim = 28 * 28
ldim = 128
cdim = 10


class LossAdd(Layer):
    def __init__(self, lossv=None, lossf=None, *args, **kwargs):
        self._lossv = lossv
        self._lossf = lossf
        super().__init__(*args, **kwargs)

    def build(self, input_shape, **kwargs):
        if not self._lossv is None:
            self.add_loss(self._lossv)
        super().build(input_shape, **kwargs)

    def call(self, in_t):
        if self._lossf:
            self.add_loss(self._lossf(in_t), inputs=in_t)
        return in_t


gl_in = Input(batch_shape=(bsize, ldim))
gc_in = Input(batch_shape=(bsize, cdim))
c_emb = Dense(16)(gc_in)
x = Concatenate()([gl_in, gc_in])
x = Dense(256)(x)
x = LeakyReLU()(x)
x = Dense(1024)(x)
x = LeakyReLU()(x)
x = Reshape((2, 2, 256))(x)

y = x
x = Conv2D(128, (1, 1))(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(128, (3, 3), use_bias=False, strides=(2, 2), padding='same')(x)
z = Dense(128)(c_emb)
x = Lambda(lambda els: els[0] + els[1][:, None, None, :])([x, z])
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(256, (1, 1))(x)
y = UpSampling2D()(y)
x = Add()([x, y])
x = BatchNormalization()(x)
x = LeakyReLU()(x)

y = x
x = Conv2D(64, (1, 1))(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(128, (3, 3), use_bias=False, strides=(2, 2), padding='same')(x)
z = Dense(128)(c_emb)
x = Lambda(lambda els: els[0] + els[1][:, None, None, :])([x, z])
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(128, (1, 1))(x)
y = Conv2D(128, (1, 1))(y)
y = UpSampling2D()(y)
x = Add()([x, y])
x = BatchNormalization()(x)
x = LeakyReLU()(x)

y = x
x = Conv2D(64, (1, 1))(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(128, (3, 3), use_bias=False, padding='same')(x)
z = Dense(128)(c_emb)
x = Lambda(lambda els: els[0] + els[1][:, None, None, :])([x, z])
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(128, (1, 1))(x)
x = Add()([x, y])
x = BatchNormalization()(x)
x = LeakyReLU()(x)

y = x
x = Conv2D(64, (1, 1))(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
z = Dense(64)(c_emb)
x = Lambda(lambda els: els[0] + els[1][:, None, None, :])([x, z])
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(128, (1, 1))(x)
y = UpSampling2D()(y)
x = Add()([x, y])
x = BatchNormalization()(x)
x = LeakyReLU()(x)

y = x
x = Conv2D(64, (1, 1))(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(64, (3, 3), use_bias=False, padding='same')(x)
z = Dense(64)(c_emb)
x = Lambda(lambda els: els[0] + els[1][:, None, None, :])([x, z])
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(128, (1, 1))(x)
x = Add()([x, y])
x = BatchNormalization()(x)
x = LeakyReLU()(x)

y = x
x = Conv2D(64, (1, 1))(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
z = Dense(32)(c_emb)
x = Lambda(lambda els: els[0] + els[1][:, None, None, :])([x, z])
x = LeakyReLU()(x)
x = Conv2D(64, (1, 1))(x)
y = Conv2D(64, (1, 1))(y)
y = UpSampling2D()(y)
x = Add()([x, y])
x = LeakyReLU()(x)

x = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
g_out = x

gen_model = Model([gl_in, gc_in], g_out)
print(gen_model.summary())

dd_in = Input(batch_shape=(bsize, 32, 32, 3))
dc_in = Input(batch_shape=(bsize, cdim))
c_emb = Dense(10)(dc_in)

x = dd_in
x = Conv2D(64, (3, 3), padding='same')(x)
y = Dense(64)(c_emb)
x = Lambda(lambda els: els[0] + els[1][:, None, None, :])([x, y])
x = LeakyReLU()(x)
x = MaxPool2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), padding='same')(x)
y = Dense(64)(c_emb)
x = Lambda(lambda els: els[0] + els[1][:, None, None, :])([x, y])
x = LeakyReLU()(x)
x = MaxPool2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), padding='same')(x)
y = Dense(128)(c_emb)
x = Lambda(lambda els: els[0] + els[1][:, None, None, :])([x, y])
x = LeakyReLU()(x)
x = MaxPool2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(256, (3, 3), padding='same')(x)
y = Dense(256)(c_emb)
x = Lambda(lambda els: els[0] + els[1][:, None, None, :])([x, y])
x = LeakyReLU()(x)
x = MaxPool2D(pool_size=(2, 2), padding='same')(x)
x = Flatten()(x)

x = Dense(64)(x)
x = LeakyReLU()(x)
x = Dense(1)(x)
d_out = x

disc_model = Model([dd_in, dc_in], d_out)
print(disc_model.summary())

adv_model = Model([dd_in, dc_in], d_out)
adv_model.trainable = False
l_in = Input(batch_shape=(bsize, ldim))
c_in = Input(batch_shape=(bsize, cdim))
gen_out = gen_model([l_in, c_in])
adv_out = adv_model([gen_out, c_in])
gan_out = LossAdd(lossf=lambda x: -K.mean(x))(adv_out)
gan_model = Model([l_in, c_in], gan_out)
print(gan_model.summary())

est_model = Model([gl_in, gc_in], g_out)
est_model.trainable = False
nin = Input(batch_shape=(bsize, ldim))
rin = Input(batch_shape=(bsize, 32, 32, 3))
ec_in = Input(batch_shape=(bsize, cdim))
cc_in = Input(batch_shape=(bsize, cdim))

fin = est_model([nin, ec_in])
allin = Concatenate(axis=0)([rin, fin])
c_in = Concatenate(axis=0)([cc_in, ec_in])


def get_hin(lst):
    rr, cc, ff, ec = lst
    epln = tf.random_uniform((tf.shape(rin)[0],), minval=0., maxval=1.)
    epln1, epln2 = epln[:, None, None, None], epln[:, None]
    hin = tf.stop_gradient(epln1 * rr + (1. - epln1) * ff)
    hc_in = tf.stop_gradient(epln2 * cc + (1. - epln2) * ec)
    return [hin, hc_in]


hin, hc_in = Lambda(get_hin)([rin, cc_in, fin, ec_in])
allout = disc_model([allin, c_in])
rout = Lambda(lambda aout: aout[:bsize])(allout)
fout = Lambda(lambda aout: aout[bsize:])(allout)
hout = disc_model([hin, hc_in])

lmda = 10.
gg1, gg2 = tf.gradients(hout, [hin, hc_in])
gg1 = tf.reshape(gg1, (bsize, -1))
gg2 = tf.reshape(gg2, (bsize, -1))
ttn = tf.sqrt(tf.norm(gg1 + 1e-32, axis=-1) ** 2 +
              tf.norm(gg2 + 1e-32, axis=-1) ** 2)
gradient_loss = lmda * tf.reduce_mean((ttn - 1.) ** 2)

rout = LossAdd(lossf=lambda x: -K.mean(x))(rout)
fout = LossAdd(lossf=lambda x: K.mean(x))(fout)
hout = LossAdd(lossv=gradient_loss)(hout)

crit_model = Model([rin, cc_in, nin, ec_in], [rout, fout, hout])
print(crit_model.summary())

real_gen = gen_samp(bsize)
optgan = Adam(lr=0.2e-4, beta_1=0., beta_2=.9)
optcrit = Adam(lr=0.2e-4, beta_1=0., beta_2=.9)

gan_model.compile(optgan, loss=None)
crit_model.compile(optgan, loss=None)

gen_model.load_weights('gen_cifar10_v6_res.h5')
disc_model.load_weights('disc_cifar10_v6_res.h5')

ncrit = 7
lgan = 0.
lcrit = 0.
tmr = TicToc()
tmr.tic()


def getnnc():
    nf = np.random.normal(size=(bsize, ldim))
    cf = np.zeros((bsize, cdim))
    cf[np.arange(bsize), np.random.randint(0, cdim, size=(bsize,))] = 1.
    return nf, cf


for i in range(50000):
    for j in range(ncrit):
        loss = -crit_model.train_on_batch([*next(real_gen), *getnnc()], None)
        lcrit = .99 * lcrit + .01 * loss

    loss = gan_model.train_on_batch([*getnnc()], None)
    lgan = .99 * lgan + .01 * loss

    if not i % 100:
        print('Step=%6d | critic_loss=%0.6f | generator_loss = %0.6f | Took: %0.2fs'
              % (i, lcrit, lgan, tmr.tocvalue(True)))

    if not i % 2500:
        gen_model.save('gen_cifar10_temp.h5')
        disc_model.save('disc_cifar10_temp.h5')
        plt.figure()
        nf = np.random.normal(size=(bsize, ldim))
        cf = np.zeros((bsize, 10))
        cf[np.arange(bsize), ([*np.arange(10)] * 7)[:bsize]] = 1.
        fakes = gen_model.predict_on_batch([nf, cf])
        for k in range(32):
            plt.subplot(4, 8, 1 + k)
            plt.axis('off')
            plt.imshow(fakes[k], cmap='gray')

        # plt.tight_layout()
        plt.pause(.01)
