# -*- coding: utf-8 -*-
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
from keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
from keras.callbacks import *


K.clear_session()

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_trn = X_train.reshape((X_train.shape[0],-1))
X_tst = X_test.reshape((X_test.shape[0],-1))

def sigtan(x:tf.Tensor):
    feat_len = x.shape.as_list()[-1]//2
    xsig=x[...,:feat_len]
    xtan=x[...,feat_len:]
    return K.sigmoid(xsig)*K.tanh(xtan)

def g_act(x):
    return Lambda(sigtan)(x)

in_t = Input((28*28,))
x = in_t
x = Embedding(256, 64)(x)
x = Reshape((28,28, -1))(x)

x = ZeroPadding2D(((1,0),(1,0)))(x)

xh = Lambda(lambda v: v[:,1:,:-1,:])(x)
xv = Lambda(lambda v: v[:,:-1,1:,:])(x)

yh = xh
yv = xv
xh = ZeroPadding2D(((0,0),(4,0)))(xh)
xh = Conv2D(128, (1,5))(xh)
xv = ZeroPadding2D(((4,0),(2,2)))(xv)
xv = Conv2D(128, (5,5))(xv)
vc = Conv2D(128, (1,1))(xv)
xh = Add()([xh, vc])
xh = g_act(xh)
xv = g_act(xv)
xh = Conv2D(64, (1,1))(xh)
#xh = Add()([xh, yh])
xh = Concatenate()([xh, yh])
xv = Concatenate()([xv, yv])

yh = xh
#yv = xv
xh = ZeroPadding2D(((0,0),(4,0)))(xh)
xh = Conv2D(128, (1,5))(xh)
xv = ZeroPadding2D(((4,0),(2,2)))(xv)
xv = Conv2D(128, (5,5))(xv)
vc = Conv2D(128, (1,1))(xv)
xh = Add()([xh, vc])
xh = BatchNormalization()(xh)
xv = BatchNormalization()(xv)
xh = g_act(xh)
xv = g_act(xv)
xh = Conv2D(128, (1,1))(xh)
xh = Add()([xh, yh])
#xh = Concatenate()([xh, yh])
#xv = Concatenate()([xv, yv])

yh = xh
#yv = xv
xh = ZeroPadding2D(((0,0),(2,0)))(xh)
xh = Conv2D(128, (1,3))(xh)
xv = ZeroPadding2D(((2,0),(1,1)))(xv)
xv = Conv2D(256, (3,3))(xv)
vc = Conv2D(128, (1,1))(xv)
xh = Add()([xh, vc])
xh = BatchNormalization()(xh)
xv = BatchNormalization()(xv)
xh = g_act(xh)
xv = g_act(xv)
xh = Conv2D(128, (1,1))(xh)
xh = Add()([xh, yh])
#xh = Concatenate()([xh, yh])
#xv = Concatenate()([xv, yv])

yh = xh
#yv = xv
xh = ZeroPadding2D(((0,0),(4,0)))(xh)
xh = Conv2D(128, (1,3), dilation_rate=(1,2))(xh)
xv = ZeroPadding2D(((4,0),(2,2)))(xv)
xv = Conv2D(256, (3,3), dilation_rate=(2,2))(xv)
vc = Conv2D(128, (1,1))(xv)
xh = Add()([xh, vc])
xh = BatchNormalization()(xh)
xv = BatchNormalization()(xv)
xh = g_act(xh)
xv = g_act(xv)
xh = Conv2D(128, (1,1))(xh)
xh = Add()([xh, yh])
#xh = Concatenate()([xh, yh])
#xv = Concatenate()([xv, yv])

yh = xh
#yv = xv
xh = ZeroPadding2D(((0,0),(4,0)))(xh)
xh = Conv2D(128, (1,3), dilation_rate=(1,2))(xh)
xv = ZeroPadding2D(((4,0),(2,2)))(xv)
xv = Conv2D(256, (3,3), dilation_rate=(2,2))(xv)
vc = Conv2D(128, (1,1))(xv)
xh = Add()([xh, vc])
xh = BatchNormalization()(xh)
xv = BatchNormalization()(xv)
xh = g_act(xh)
xv = g_act(xv)
xh = Conv2D(128, (1,1))(xh)
xh = Add()([xh, yh])
#xh = Concatenate()([xh, yh])
#xv = Concatenate()([xv, yv])

yh = xh
#yv = xv
xh = ZeroPadding2D(((0,0),(4,0)))(xh)
xh = Conv2D(128, (1,3), dilation_rate=(1,2))(xh)
xv = ZeroPadding2D(((4,0),(2,2)))(xv)
xv = Conv2D(256, (3,3), dilation_rate=(2,2))(xv)
vc = Conv2D(128, (1,1))(xv)
xh = Add()([xh, vc])
xh = BatchNormalization()(xh)
xv = BatchNormalization()(xv)
xh = g_act(xh)
xv = g_act(xv)
xh = Conv2D(128, (1,1))(xh)
xh = Add()([xh, yh])
#xh = Concatenate()([xh, yh])
#xv = Concatenate()([xv, yv])

yh = xh
yv = xv
xh = ZeroPadding2D(((0,0),(4,0)))(xh)
xh = Conv2D(256, (1,3), dilation_rate=(1,2))(xh)
xv = ZeroPadding2D(((4,0),(2,2)))(xv)
xv = Conv2D(256, (3,3), dilation_rate=(2,2))(xv)
vc = Conv2D(256, (1,1))(xv)
xh = Add()([xh, vc])
xh = BatchNormalization()(xh)
xv = BatchNormalization()(xv)
xh = g_act(xh)
xv = g_act(xv)
xh = Conv2D(128, (1,1))(xh)
#xh = Add()([xh, yh])
xh = Concatenate()([xh, yh])
xv = Concatenate()([xv, yv])

yh = xh
#yv = xv
xh = ZeroPadding2D(((0,0),(4,0)))(xh)
xh = Conv2D(256, (1,3), dilation_rate=(1,2))(xh)
xv = ZeroPadding2D(((4,0),(2,2)))(xv)
xv = Conv2D(512, (3,3), dilation_rate=(2,2))(xv)
vc = Conv2D(256, (1,1))(xv)
xh = Add()([xh, vc])
xh = BatchNormalization()(xh)
xv = BatchNormalization()(xv)
xh = g_act(xh)
xv = g_act(xv)
xh = Conv2D(256, (1,1))(xh)
#xh = Add()([xh, yh])
xh = Concatenate()([xh, yh])
xv = Concatenate()([xv, yv])

yh = xh
#yv = xv
xh = ZeroPadding2D(((0,0),(4,0)))(xh)
xh = Conv2D(512, (1,3), dilation_rate=(1,2))(xh)
xv = ZeroPadding2D(((4,0),(2,2)))(xv)
xv = Conv2D(512, (3,3), dilation_rate=(2,2))(xv)
vc = Conv2D(512, (1,1))(xv)
xh = Add()([xh, vc])
xh = g_act(xh)
xv = g_act(xv)
xh = Conv2D(512, (1,1))(xh)
xh = Add()([xh, yh])
#xh = Concatenate()([xh, yh])
#xv = Concatenate()([xv, yv])


x = Concatenate()([xh, xv])
x = Reshape((28*28, -1))(x)
x = Conv1D(256, 1, activation='softmax')(x)
out_t = x

model = Model(in_t, out_t)
print(model.summary())


opt = Adam(lr = 5e-5)
    
model.compile(opt, loss=sparse_categorical_crossentropy)

mcp = ModelCheckpoint('pcnnpv2dbn.h5')
#model.fit(X_trn, X_trn[...,None], batch_size=64, 
#          epochs=50, validation_data=(X_tst,X_tst[...,None]), callbacks=[mcp])
model.load_weights('pcnnpv2dbn.h5')

nsample = 32
Samp = np.zeros((nsample,28*28))
for k in range(0,28*28-1):
    if not k%30:
        print('{}/783'.format(k))
    probs = model.predict(Samp)[:,k,:].reshape(nsample,256)
#    probs[probs<1e-3]=0.
#    probs = probs/probs.sum(axis=-1)[:,None]
    newel = np.argmax(np.log(probs)+np.random.gumbel(size=(nsample,256)), axis=-1)
    Samp[:,k]=newel
    
Samp = Samp.reshape(-1,28,28)

plt.figure()
for k in range(nsample):
    plt.subplot(4,8,1+k)
    plt.axis('off')
    plt.imshow(Samp[k], cmap='gray')



