# utils
import numpy as np
from glob import glob
import os
import warnings
from collections import namedtuple
os.environ['PYTHONHASHSEED']="0"
np.random.seed(0)

FOLDER = os.getcwd() + "/"
INPUT = (28, 28)

Label = namedtuple("Label", ['veg', 'day', 'temp', 'repl'])


def filter_images_and_labels(cond, images_and_labels_iter):
    return list(filter(lambda im_label: cond(im_label[1]), images_and_labels_iter))

def load_images_and_labels(prefix=""):
    files = list(glob(FOLDER + "data/flat_files/%s*.csv" % prefix))
    results = []
    for i, file in enumerate(files):
        M = np.loadtxt(file, delimiter=",")
        label_sting = os.path.splitext(os.path.basename(file))[0]
        try:
            # if the data isn't in the right format, or the user doesn't care about labels.
            label = Label(*label_sting.split("-"))
        except:
            warnings.warn("Skipping label extraction. Modify `Label` in scr/utils.py if desired.")
        results.append((M, label))
    return results

def collapse_images_to_single_matrix(images_and_labels):
    n = len(images_and_labels)
    M = np.empty((n,) + INPUT)
    for i, (image, label) in enumerate(images_and_labels):
        M[i, :] = image

    return np.expand_dims(M, axis=3), [_[1] for _ in images_and_labels]

def remove_rayleigh(images_and_labels):
    return [(remove_rayleigh_scatter_from_image(img), label) for img, label in images_and_labels]


def split_images_and_labels(images_and_labels, test_frac=0.05, just_images=False):
    images_and_labels = list(images_and_labels)
    n = len(images_and_labels)
    n_train_samples = int((1-test_frac) * n)
    ix = np.arange(n)
    np.random.shuffle(ix)
    if just_images:
        return [images_and_labels[i][0] for i in ix[:n_train_samples]],\
                [images_and_labels[i][0] for i in ix[n_train_samples:]]
    else:
        return [images_and_labels[i] for i in ix[:n_train_samples]],\
                [images_and_labels[i] for i in ix[n_train_samples:]]


def kth_diag_indices(a, k):
    rowidx, colidx = np.diag_indices_from(a)
    colidx = colidx.copy()  # rowidx and colidx share the same buffer

    if k > 0:
        colidx += k
    else:
        rowidx -= k
    k = np.abs(k)

    return rowidx[:-k], colidx[:-k]


def remove_rayleigh_scatter_from_image(img):
    for ix in np.arange(5, 28):
        img[kth_diag_indices(img, ix)] = 0

    return img
