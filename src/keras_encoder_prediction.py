import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, BatchNormalization
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from src.utils import FOLDER, load_images, INPUT

def load_encoder():
    encoder = keras.models.load_model(FOLDER + "trained_models/simple_keras_conv_encoder.h5")
    return encoder

def load_decoder():
    decoder = keras.models.load_model(FOLDER + "trained_models/simple_keras_conv_decoder.h5")
    return decoder


#images = concat_and_expand(load_images())
images = load_images()
encoder = load_encoder()
encoder.summary()

decoder = load_decoder()
decoder.summary()

prediction = encoder.predict(images)


prediction_images = decoder.predict(prediction)

for _id in np.arange(1, 200, 40):
    fig = plt.figure(figsize=(6, 8))
    axes = fig.subplots(3, 1)

    axes[0].imshow(images[_id].reshape(*INPUT))
    axes[0].set_title("Actual")

    axes[1].imshow(prediction[_id][None, :], cmap=cm.Greys)
    axes[1].set_title("Latent representation")

    axes[2].imshow(prediction_images[_id].reshape(*INPUT))
    axes[2].set_title("Reconstructed")


plt.show(block=True)
