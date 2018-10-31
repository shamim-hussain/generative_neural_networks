from keras.datasets import mnist
from keras.layers import *
from keras.optimizers import Adam
from keras.models import *
from keras import backend as K
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pytictoc import TicToc
from keras.utils import to_categorical

K.clear_session()

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#Xf_train = X_train.reshape((X_train.shape[0], -1))

bsize= 64

def gen_samp(X, bsize):
    lx = len(X)
    Xf = X.reshape((X.shape[0], -1))
    ind = np.arange(lx)
    while True:
        np.random.shuffle(ind)
        for k in range(0, lx, bsize):
            ss = np.arange(k,min(k+bsize, lx))
            if k+bsize>lx:
                ss = np.append(ss, np.arange(k+bsize-lx))
            Xs = Xf[ind[ss]]
            yield Xs[:,:-1], Xs[:,1:, None]


in_t = Input((None,))
x = in_t
x = Embedding(256, 64)(x)

y = x
x = Conv1D(64, 3, padding='causal')(x)
x = Conv1D(128, 3, padding='causal', dilation_rate=26)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Concatenate()([x,y])

y = x
x = Conv1D(64, 3, padding='causal', dilation_rate=2)(x)
x = Conv1D(128, 3, padding='causal', dilation_rate=55)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Concatenate()([x,y])

y = x
x = CuDNNLSTM(512, return_sequences=True)(x)
x = Concatenate()([x,y])

y = x
x = Conv1D(64, 3, padding='causal')(x)
x = Conv1D(64, 3, padding='causal', dilation_rate=26)(x)

y = Conv1D(64, 1)(y)
x = Add()([x, y])
x = LeakyReLU()(x)

x = Conv1D(256, 1, activation='softmax')(x)
out_t = x

model = Model(in_t, out_t)
print(model.summary())

opt = Adam(lr = 5e-5)
model.compile(opt, loss='sparse_categorical_crossentropy')

train_gen = gen_samp(X_train, bsize)
val_gen = gen_samp(X_test, bsize)
#model.fit_generator(train_gen, len(X_train)//bsize, epochs=50, 
#                    validation_data=val_gen,
#                    validation_steps=len(X_test)//bsize)
model.load_weights('ar_lstm_v3.h5')

nsample = 32
Samp = np.zeros((nsample,28*28))
for k in range(1,28*28):
    if not k%30:
        print('{}/783'.format(k))
    probs = model.predict(Samp[:,:k])[:,-1,:].reshape(32,256)
    probs[probs<1e-3]=0.
    probs = probs/probs.sum(axis=-1)[:,None]
    newel = np.argmax(np.log(probs)+np.random.gumbel(size=(nsample,256)), axis=-1)
    Samp[:,k]=newel
    
Samp = Samp.reshape(-1,28,28)

plt.figure()
for k in range(32):
    plt.subplot(4,8,1+k)
    plt.axis('off')
    plt.imshow(Samp[k], cmap='gray')
