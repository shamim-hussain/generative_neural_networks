from keras.datasets import mnist
from keras.layers import *
from keras.optimizers import *
from keras.models import *
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from custom_layers import *

plt.close('all')
K.clear_session()

batch_size = 256
num_epochs = 50
steps_per_epoch = 40
latent_dim = 2

num_bins = 4
sigma = .5

d_w = 2*np.pi/num_bins
i = np.arange(num_bins)
rad = float(num_bins*2)
centroids = np.array([np.cos(d_w*i)*rad, np.sin(d_w*i)*rad]).T




(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train[..., None] / 255.
X_test = X_test[..., None] / 255.

#train_pos = np.nonzero((Y_train==0)|(Y_train==1)|(Y_train==6)|(Y_train==9))
#test_pos = np.nonzero((Y_test==0)|(Y_test==1)|(Y_test==6)|(Y_test==9))
#X_train = X_train[train_pos]
#Y_train = Y_train[train_pos]
#X_test = X_test[test_pos]
#Y_test = Y_test[test_pos]

X = X_train.astype(np.float32)

# =======Encoder==========
x = Input((28, 28, 1))
enc_in = x
x = Conv2D(16, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)
x = Dropout(.2)(x)
x = Conv2D(16, (3, 3), strides=(2, 2), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)
x = Dropout(.2)(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)
x = Conv2D(64, (3, 3),padding='same')(x)
x = LeakyReLU(alpha=.2)(x)
x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)

x = Conv2D(64, (5, 5),  padding='same')(x)
x = LeakyReLU(alpha=.2)(x)
x = Conv2D(128, (5, 5), strides=(4, 4),  padding='same')(x)
x = LeakyReLU(alpha=.2)(x)

x = Flatten()(x)
x = Dense(128)(x)
x = LeakyReLU(alpha=.2)(x)
x = Dense(latent_dim)(x)
enc_out = x

# =======Decoder==========
x = Input((latent_dim,))
dec_in = x
x = Dense(128)(x)
x = LeakyReLU(alpha=.2)(x)
x = Dense(512)(x)
x = LeakyReLU(alpha=.2)(x)
x = Reshape((2, 2, 128))(x)

x = UpSampling2D((3, 3))(x)
x = ZeroPadding2D(((1, 0), (1, 0)))(x)
x = Conv2D(64, (5, 5), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)
x = Conv2D(64, (5, 5), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (5, 5), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.2)(x)
x = Conv2D(1, (3, 3), padding='same')(x)
dec_out = x

# =======Discriminator==========
x = Input((latent_dim,))
disc_in = x
x = Dense(32)(x)
x = LeakyReLU(alpha=.2)(x)
x = Dropout(.1)(x)
x = Dense(64)(x)
x = LeakyReLU(alpha=.2)(x)
x = Dropout(.3)(x)
x = Dense(32)(x)
x = LeakyReLU(alpha=.2)(x)
x = Dense(1, activation='sigmoid')(x)
disc_out = x

# =============Models==============
disc_model = Model(disc_in, disc_out)
disc_fixed = Model(disc_in, disc_out)
disc_fixed.trainable = False

enc_model = Model(enc_in, enc_out)
dec_model = Model(dec_in, dec_out)

adv_loss, adv_out = AdvLossLayer(0.01)(disc_fixed(enc_out))
ae_out = MAELossLayer(2.)([dec_model(enc_out), enc_in])
aae_model = Model([enc_in], [ae_out, adv_out])

opt_aae = Adam(lr=5e-4)
opt_disc = Adam(lr=5e-4)

aae_model.compile(optimizer=opt_aae, loss=None)
disc_model.compile(optimizer=opt_disc, loss='binary_crossentropy', metrics=['acc'])
print(disc_model.summary())
print(aae_model.summary())

X_ind = np.arange(0, len(X))

lmbda = .95
d_scores = [0.] * 2
aae_score = 0.


def upd_scores(old, new):
    for i in range(len(old)):
        old[i] = lmbda * old[i] + (1 - lmbda) * new[i]


for e in range(num_epochs):
    print('\n*****************')
    print('Epoch:', e)
    np.random.shuffle(X_ind)
    X = X[X_ind]
    for s in range(0, len(X), batch_size):
        if not (s // batch_size + 1) % steps_per_epoch:
            print('===================')
            print('Step:', (s // batch_size + 1))
            print('-------------------')
            print('Disc_loss:', d_scores[0])
            print('Disc_acc:', d_scores[1] * 100)
            print('AAE_loss:', aae_score)

        end_ind = min(s + batch_size, len(X))
        reals = X[s:end_ind]

        aae_score = lmbda * aae_score + (1 - lmbda) * aae_model.train_on_batch(reals, None)

        fakes = enc_model.predict_on_batch(reals)
        
        noises = []
        for i in range(num_bins):
            noises.append(np.random.randn((len(fakes)+i)//num_bins,
                                          latent_dim)*sigma + centroids[i])
        noise = np.concatenate(noises)

        assert len(fakes) == len(noise)
        code_batch = np.concatenate((noise, fakes))
        lbl_batch = np.concatenate((np.ones(len(noise)), np.zeros(len(fakes))))

        assert len(code_batch) == len(lbl_batch)
        Y_ind = np.arange(0, len(code_batch))
        np.random.shuffle(Y_ind)
        code_batch = code_batch[Y_ind]
        lbl_batch = lbl_batch[Y_ind]

        upd_scores(d_scores, disc_model.train_on_batch(code_batch, lbl_batch))


    x_test_encoded = enc_model.predict(X_test, batch_size=batch_size)
    f1 = plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=Y_test)
    plt.colorbar()
    plt.show()
    
    
    
    # display a 2D manifold of the digits
    n = 20  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    
    noises = []
    for i in range(num_bins):
        noises.append(np.random.randn((n+i)//num_bins, n,
                                      latent_dim)*sigma + centroids[i])
    noise = np.concatenate(noises)
    for i in range(n):
        for j in range(n):
            z_sample = noise[i, j][None, ...]
            x_decoded = dec_model.predict(z_sample)
            digit = x_decoded[0,:,:,0]
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    
    f2 = plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
    
    
    f1.savefig('f1_v3.png')
    f2.savefig('f2_v3.png')
    
    enc_model.save('enc_v3.hdf5')
    dec_model.save('dec_v3.hdf5')


