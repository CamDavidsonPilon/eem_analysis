# script to move matlab files to (30, 25) flat files
import glob
import numpy as np
from scipy import io
from skimage.transform import resize



FOLDER = "/Users/camerondavidson-pilon/code/eem_conv_autoencoder/"
OUTPUT_SIZE = (28, 28)


data_blob = dict()
for file in glob.glob(FOLDER + "data/raw_data_as_matlab_files/Data-*.mat"):
    _data_blob = io.loadmat(file)
    veggie = list(_data_blob.keys())[3]

    # tensor is indexed: experimental_conditions, emission, excitation

    data_blob[veggie] = _data_blob[veggie]


for veggie in ['spinach', 'celery', 'kale', 'cucumber']:
    for i in range(data_blob[veggie].shape[0]):
        M = data_blob[veggie][i]
        # reshape M to OUTPUT_SIZE
        M_reshaped = resize(M, OUTPUT_SIZE, preserve_range=True)

        # scale to be between 0 and 1
        M_reshaped = M_reshaped / M_reshaped.max()

        # are there any nans?
        assert np.isnan(M_reshaped).sum() == 0

        np.savetxt(FOLDER + "data/flat_files/%s-%s.csv" % (veggie, str(i).zfill(2)), M_reshaped, delimiter=',')

