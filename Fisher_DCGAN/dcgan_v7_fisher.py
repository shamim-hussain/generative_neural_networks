

from keras.datasets import mnist
from keras.layers import *
from keras.optimizers import RMSprop, Adam
from keras.models import *
from keras import backend as K
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pytictoc import TicToc

K.clear_session()

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.

bsize=64

def gen_samp(bsize):
    lx = len(X_train)
    ind = np.arange(lx)
    while True:
        np.random.shuffle(ind)
        for k in range(0, lx, bsize):
            ss = np.arange(k,min(k+bsize, lx))
            if k+bsize>lx:
                ss = np.append(ss, np.arange(k+bsize-lx))
            yield X_train[ind[ss]]

sdim = 28*28
ldim = 32


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
        

gen_model = Sequential()
gen_model.add(Dense(32, input_shape=(ldim,)))
#gen_model.add(Dense(32, batch_input_shape=(bsize,ldim)))
gen_model.add(LeakyReLU())
gen_model.add(Dense(1024))
gen_model.add(LeakyReLU())
gen_model.add(Reshape((2,2,256)))
gen_model.add(Conv2DTranspose(128, (3,3), use_bias=False, strides=(2,2)))
gen_model.add(BatchNormalization())
gen_model.add(LeakyReLU())
gen_model.add(Conv2DTranspose(128, (4,4), use_bias=False,  strides=(2,2)))
gen_model.add(BatchNormalization())
gen_model.add(LeakyReLU())
gen_model.add(Conv2DTranspose(64, (3,3),strides=(2,2)))
gen_model.add(LeakyReLU())
gen_model.add(Conv2DTranspose(1, (4,4), activation = 'sigmoid'))
gen_model.add(Reshape((28,28)))

print(gen_model.summary())

disc_model = Sequential()
disc_model.add(Reshape((28,28,1),input_shape=(28,28)))
#disc_model.add(Reshape((28,28,1),batch_input_shape=(bsize,28,28)))
disc_model.add(Conv2D(64, (3,3)))
disc_model.add(LeakyReLU())
disc_model.add(MaxPool2D(pool_size=(2,2), padding='same'))
disc_model.add(Conv2D(64, (3,3)))
disc_model.add(LeakyReLU())
disc_model.add(MaxPool2D(pool_size=(2,2), padding='same'))
disc_model.add(Conv2D(128, (3,3)))
disc_model.add(LeakyReLU())
disc_model.add(MaxPool2D(pool_size=(2,2), padding='same'))
disc_model.add(Flatten())
disc_model.add(Dense(32))
disc_model.add(LeakyReLU())
disc_model.add(Dense(1))
print(disc_model.summary())


adv_model = Model(disc_model.input, disc_model.output)
adv_model.trainable = False
gan_model = Sequential()
gan_model.add(gen_model)
gan_model.add(adv_model)
gan_model.add(LossAdd(lossf=lambda x: -K.mean(x)))
print(gan_model.summary())


#rin = Input((28,28))
#fin = Input((28,28))
rin = Input(batch_shape=(bsize,28,28))
fin = Input(batch_shape=(bsize,28,28))

allin = Concatenate(axis=0)([rin,fin])
allout = disc_model(allin)
    
rout = Lambda(lambda x: x[:bsize])(allout)
fout = Lambda(lambda x: x[bsize:])(allout)

#rout = LossAdd(lossf=lambda x: -K.mean(x))(rout)
#fout = LossAdd(lossf=lambda x: K.mean(x))(fout)
wdist = K.mean(fout)-K.mean(rout)
hout = Lambda(lambda x: K.mean(x[0]**2)-1.)([allout, rout, fout])

rho = 1e-6
llambda = K.variable(0.)

#def update_lambda(x):
#    with tf.control_dependencies([tf.assign(llambda, llambda + rho*x)]):
#        xd = tf.identity(x)
#    return xd
#def hloss(x):
#    mv = K.mean(x)
#    sv = mv**2
#    return llambda*mv + (rho/2)*sv

#hout = LossAdd(lossf=hloss)(hout)

hout = LossAdd(lossf=lambda x: llambda*x + (rho/2)*(x**2)+wdist)(hout)#
#hout = Lambda(update_lambda)(hout)

crit_model=Model([rin, fin], [hout])
print(crit_model.summary())


real_gen = gen_samp(bsize)
#optgan = Adam(lr=5e-5, beta_1=0., beta_2=.9)
#optcrit = Adam(lr=5e-5, beta_1=0., beta_2=.9)

optgan = Adam(lr=1e-4)#, beta_1=0., beta_2=.9)
optcrit = Adam(lr=1e-4)#, beta_1=0., beta_2=.9)

gan_model.compile(optgan, loss=None)
crit_model.compile(optgan, loss=None)
crit_model.metrics_tensors.extend([hout, llambda, wdist])
#crit_model.add_update(tf.assign(llambda, llambda + rho*hout))

gen_model.load_weights('gen_fisher.h5')
disc_model.load_weights('disc_fisher.h5')

ncrit = 2
lgan = 0.
lcrit = 0.
lnorm = 0.
tmr = TicToc()
tmr.tic()
for i in range (150000):
    fakes = gen_model.predict_on_batch(np.random.normal(size=(bsize*ncrit, ldim)))
    for j in range(ncrit):
        reals = next(real_gen)
        loss, hv, lv, wd = crit_model.train_on_batch([reals, fakes[j*bsize:(j+1)*bsize]], None)
        lcrit = .99*lcrit-.01*wd
        lnorm = .99*lnorm+.01*hv
        K.set_value(llambda, lv+rho*hv)
    
    loss = gan_model.train_on_batch(np.random.normal(size=(bsize, ldim)), None)
    lgan = .99*lgan+.01*loss
    
    if not i%100:
        print('Step=%6d | critic_loss=%0.5f | norm_loss = %0.2f | generator_loss = %0.5f | Took: %0.2fs'
              %(i,lcrit, lnorm, lgan, tmr.tocvalue(True)))
    
    if not i%5000:
        gen_model.save('gen_fisher_temp.h5')
        disc_model.save('disc_fisher_temp.h5')
        plt.figure(figsize=(15,5))
        for k in range(60):
            plt.subplot(6,10,1+k)
            plt.imshow(fakes[k], cmap='gray')
        plt.pause(.01)
        figs=plt.get_fignums()
        if len(figs)>15: 
            for ff in figs[0:len(figs)-15]: plt.close(ff)
