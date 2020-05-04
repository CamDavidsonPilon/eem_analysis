import tensorly as tl
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt
from matplotlib import cm
from keras.utils import plot_model

from src.utils import *

train_images_and_labels, _ = split_images_and_labels(load_images_and_labels())
train_images, _ = collapse_images_to_single_matrix(train_images_and_labels)
LATENT_DIM_SIZE = 12


train_images =  train_images[:,:,:,0]


results = parafac(train_images, rank=LATENT_DIM_SIZE)

prediction = results.factors
prediction_images = tl.kruskal_to_tensor(results)

for _id in np.arange(1, 200, 40):
    fig = plt.figure(figsize=(6, 8))
    axes = fig.subplots(2, 1)

    axes[0].imshow(train_images[_id].reshape(*INPUT))
    axes[0].set_title("Actual")

    axes[1].imshow(prediction_images[_id].reshape(*INPUT))
    axes[1].set_title("Reconstructed")


plt.show(block=True)

# MAE - I can only compare evaluation of training images
running_sum = 0
running_count = 0
for predicted, actual in zip(prediction_images, train_images):
    running_count += 1
    running_sum += np.abs(predicted - actual).sum()


print(running_sum / running_count)
# 19.4353