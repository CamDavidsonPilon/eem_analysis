import numpy as np
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import keras
from src.utils import *
from src.keras_encoder_reconstruction import load_encoder, load_decoder
from sklearn.decomposition import PCA

images_labels = filter_images_and_labels(lambda label: label.veg != 'kale', load_images_and_labels())
images, labels = collapse_images_to_single_matrix(images_labels)

encoder = load_encoder()
encoder.summary()

prediction = encoder.predict(images)

pca = PCA(n_components=3)
transformed_decomp = pca.fit_transform(prediction)

### plot vegetables
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

color_map = {'spinach': '#348ABD', 'celery': '#7A68A6', 'cucumber': '#467821'}
colors = [color_map[label.veg] for label in labels]

ax.scatter(transformed_decomp[:, 0], transformed_decomp[:, 1], alpha=0.5, c=colors)
ax.legend(handles=[mpatches.Patch(color=c, label=v) for v,c in color_map.items()])


### plot time - not working because I think the labels are scrambled.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

days = [int(label.day.strip("d").replace("NA", "0")) for label in labels]
color_map = {0: "#fafa6e", 2: "#b5e877", 4: "#77d183", 6: "#3fb78d", 8: "#009c8f", 10: "#007f86", 12: "#1c6373", 14: "#2a4858"}
colors = [color_map[d] for d in days]

ax.scatter(transformed_decomp[:, 0], transformed_decomp[:, 1], alpha=0.5, c=colors)
ax.legend(handles=[mpatches.Patch(color=c, label=v) for v,c in color_map.items()])

