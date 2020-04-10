# script to move matlab files to (30, 25) flat files
import glob
import numpy as np
from scipy import io
from skimage.transform import resize



FOLDER = "/Users/camerondavidson-pilon/code/eem_conv_autoencoder/"
OUTPUT_SIZE = (28, 28)


data_blob = dict()
# I don't think the assumption that the matlab files follow the order of the excel file is correct..
for file in glob.glob(FOLDER + "data/raw_data_as_matlab_files/Data-*.mat"):
    _data_blob = io.loadmat(file)
    veggie = list(_data_blob.keys())[3]
    try:
        _labels = np.loadtxt(FOLDER + "data/raw_data_as_matlab_files/Labels-%s.csv" % veggie, dtype=str)
    except:
        _labels = ["NA-NA"] * _data_blob[veggie].shape[0]

    _new_labels = []
    for _, _label in groupby(_labels):
        _label = list(_label)
        for i, _ in enumerate(_label):
            _new_labels.append(_label[i] + "-%d" % i)

    # tensor is indexed: experimental_conditions, emission, excitation

    data_blob[veggie] = {'EEMs': _data_blob[veggie], 'labels': _new_labels}


for veggie in ['spinach', 'celery', 'kale', 'cucumber']:
    for i in range(data_blob[veggie]['EEMs'].shape[0]):
        M = data_blob[veggie]['EEMs'][i]
        # reshape M to OUTPUT_SIZE
        M_reshaped = resize(M, OUTPUT_SIZE, preserve_range=True)

        # scale to be between 0 and 1
        M_reshaped = M_reshaped / M_reshaped.max()

        # are there any nans?
        assert np.isnan(M_reshaped).sum() == 0

        label = data_blob[veggie]['labels'][i]
        np.savetxt(FOLDER + "data/flat_files/%s-%s.csv" % (veggie, label), M_reshaped, delimiter=',')

