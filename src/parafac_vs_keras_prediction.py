import tensorly as tl
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt
from matplotlib import cm
from keras.utils import plot_model
from src.keras_encoder_reconstruction import load_encoder, load_decoder
from src.utils import *

train_images_and_labels, _ = split_images_and_labels(load_images_and_labels())
train_images, _ = collapse_images_to_single_matrix(train_images_and_labels)
LATENT_DIM_SIZE = 12


# PARAFAC
images_for_parafac =  train_images[:,:,:,0]
results = parafac(images_for_parafac, rank=LATENT_DIM_SIZE)

prediction_images_parafac = tl.kruskal_to_tensor(results)


#CNN
encoder = load_encoder()
decoder = load_decoder()

images_for_cnn = train_images
prediction = encoder.predict(images_for_cnn)
prediction_images_cnn = decoder.predict(prediction)


_id = 12

fig = plt.figure(figsize=(6, 8))
axes = fig.subplots(3, 1)

axes[0].imshow(images_for_parafac[_id].reshape(*INPUT))
axes[0].set_title("Actual")


axes[1].imshow(prediction_images_cnn[_id].reshape(*INPUT))
axes[1].set_title("Reconstructed AE-CNN")

axes[2].imshow(prediction_images_parafac[_id].reshape(*INPUT))
axes[2].set_title("Reconstructed PARAFAC")
fig.tight_layout()

