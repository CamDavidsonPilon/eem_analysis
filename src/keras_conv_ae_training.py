import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, BatchNormalization
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import FOLDER, INPUT, load_images
from keras.utils import plot_model
from scipy.special import logit

train_images, test_images = split_images(load_images())
LATENT_DIM_SIZE = 12

# Encoder
inp = Input(INPUT + (1,))
e = Conv2D(filters=30, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inp)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(40, (3, 3), activation='relu')(e)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(40, (3, 3), activation='relu')(e)
e = Flatten()(e)
encoded = Dense(LATENT_DIM_SIZE, activation='softmax')(e)

encoder = Model(inputs=inp, outputs=encoded, name='encoder')

# Decoder
decoder_input = Input(shape=(LATENT_DIM_SIZE,))
d = Dense(49, activation='relu')(decoder_input)
d = Reshape((7, 7, 1))(d)
d = Conv2DTranspose(30, (3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(30, (3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(d)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

decoder = Model(inputs=decoder_input, outputs=decoded, name='decoder')

# Model = Encoder -> Decoder
outputs = decoder(encoder(inp))
ae = Model(inp, outputs, name='ae')

ae.summary()


filepath = FOLDER + "trained_models/simple_keras_conv_ae.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True)

# fit the model
ae.compile(optimizer="nadam", loss="mean_absolute_error")
ae.fit(train_images, train_images,
        validation_data=(test_images, test_images),
        epochs=1000,
        batch_size=100,
        callbacks=[checkpoint])


encoder.save('trained_models/simple_keras_conv_encoder.h5')
decoder.save('trained_models/simple_keras_conv_decoder.h5')

