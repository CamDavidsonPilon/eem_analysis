# utils
import numpy as np
from glob import glob
import os
os.environ['PYTHONHASHSEED']="0"
np.random.seed(0)

FOLDER = os.getcwd()
INPUT = (28, 28)


def load_images(prefix=""):
    files = list(glob(FOLDER + "data/flat_files/%s*.csv" % prefix))
    M = np.empty((len(files),) + INPUT)
    for i, file in enumerate(files):
        _M = np.loadtxt(file, delimiter=",")
        M[i, :] = _M

    return  np.expand_dims(M, axis=3)


def split_images(M, test_frac=0.05):
    n = M.shape[0]
    n_train_samples = int((1-test_frac) * n)
    ix = np.arange(n)
    np.random.shuffle(ix)
    return M[ix[:n_train_samples]], M[ix[n_train_samples:]]
