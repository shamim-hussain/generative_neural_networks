# -*- coding: utf-8 -*-

from keras.layers import *
from keras.datasets import mnist
from keras import backend as K
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import metrics
from keras.optimizers import Adam
from keras.callbacks import *
from custom_layers import *
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import os, shutil
from keras.utils import to_categorical

lat_dim = 8
batch_size = 32
epochs = 15
tb_path = 'd:/vae_tb'

K.clear_session()
# if os.path.exists(tb_path):
#     shutil.rmtree(tb_path)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.expand_dims(X_train, -1) / 255.
X_test = np.expand_dims(X_test, -1) / 255.
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
                       
TrainGen = ImageDataGenerator(rotation_range=.02, zoom_range=.05, width_shift_range=.025, height_shift_range=.025,
                              shear_range=.025)
TestGen = ImageDataGenerator()

train_gen = TrainGen.flow(X_train, Y_train, 32)
test_gen = TestGen.flow(X_test, Y_test, 32)
train_iter = ((x[0], None) for x in train_gen)
test_iter = ((x[0], None) for x in test_gen)

train_aec = ((*x, None) for x in train_gen)
test_aec = ((*x, None) for x in test_gen)

# --------------Encoder--------------------
in_t = Input(shape=(X_train.shape[1:]))
x = Conv2D(16, (3, 3), activation=None, padding='same')(in_t)
x = seq_call(x, conv_block(filters=16, dropout=.2, dilation_rate=(2, 2)))
x = seq_call(x, conv_block(filters=16, dropout=None, strides=(2, 2)))

x = ZeroPadding2D()(x)
x = seq_call(x, conv_block(filters=16, dropout=.1))
x = seq_call(x, conv_block(filters=32, dropout=.2, dilation_rate=(2, 2)))
x = seq_call(x, conv_block(filters=32, dropout=None, strides=(2, 2)))

x = seq_call(x, conv_block(filters=32, dropout=.1))
x = seq_call(x, conv_block(filters=64, dropout=.2))
x = seq_call(x, conv_block(filters=64, dropout=None, strides=(2, 2)))

x = seq_call(x, conv_block(filters=64, dropout=.1))
x = seq_call(x, conv_block(filters=128, dropout=.2))
x = seq_call(x, conv_block(filters=128, dropout=None, strides=(2, 2)))

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
h = Dense(32, activation='relu')(x)

zm = Dense(lat_dim)(h)
zs = Dense(lat_dim)(h)

z = VAESampling()([zm, zs])

# --------------Decoder---------------
dh = Dense(32, activation='relu', name='dec0')(z)
x = dh
x = Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Reshape((2, 2, 64))(x)

x = seq_call(x, deconv_block(filters=128, dropout=None, strides=(2, 2)))
x = seq_call(x, deconv_block(filters=64, dropout=None))
x = seq_call(x, deconv_block(filters=64, dropout=None))

x = seq_call(x, deconv_block(filters=64, dropout=None, strides=(2, 2)))
x = seq_call(x, deconv_block(filters=32, dropout=None))
x = seq_call(x, deconv_block(filters=32, dropout=None))

x = seq_call(x, deconv_block(filters=32, dropout=None, strides=(2, 2)))
x = seq_call(x, deconv_block(filters=16, dropout=None, dilation_rate=(2,2)))
x = Cropping2D(1)(x)
x = seq_call(x, deconv_block(filters=16, dropout=None))

x = seq_call(x, deconv_block(filters=16, dropout=None, strides=(2, 2)))
x = seq_call(x, deconv_block(filters=16, dropout=None,
                             dilation_rate=(2, 2), bn=False))
x = Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='sigmoid')(x)

x = VAERecLossLayer()([x, in_t])
out_t = x
vae = Model(in_t, out_t)
encoder = Model(in_t, zm)

d_in = Input((lat_dim,))
x = d_in
fl = [i for (i, l) in enumerate(vae.layers) if l.name == 'dec0'][0]
for i in range(fl, len(vae.layers) - 1): x = vae.layers[i](x)
d_out = x

decoder = Model(d_in, d_out)


x=h
x=Dense(32, activation='relu')(x)
x=Dense(16, activation='relu')(x)
x=Dense(10, activation='softmax')(x)
c_out= x

vaec = Model(inputs=in_t, outputs=[c_out, out_t])


optim = Adam(.0005)
# vae.compile(optimizer=optim, loss=vae_loss)
vaec.compile(optimizer=optim, loss=['categorical_crossentropy',None], metrics=['acc'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.00005)

mchk=ModelCheckpoint('cnn_vae_temp.h5', 'val_loss',save_best_only=True,period=4)
vaec.load_weights('cnn_vae.h5')
hist = vaec.fit_generator(train_aec, steps_per_epoch=X_train.shape[0] // batch_size, epochs=epochs,
                         validation_data=test_aec,
                         validation_steps=X_test.shape[0] // batch_size,
                         callbacks=[reduce_lr, mchk])

x_test_encoded = encoder.predict(X_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=np.argmax(Y_test, axis=-1))
plt.colorbar()
plt.show()

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

for i in range(n):
    for j in range(n):
        z_sample = np.expand_dims(list(np.random.normal() for i in range(lat_dim)), axis=0) * .75
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0, :, :, 0]
        figure[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()


#
