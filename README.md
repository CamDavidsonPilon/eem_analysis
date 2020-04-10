## Autoencoding EEMs


### Analysis of EEMs

EEMs (excitation emission matrices) are measurements of a sample's fluorescence intensity at varying excitation and emission wavelengths.

Traditionally, EEMs have been analyzed using linear matrix decomposition methods like PARAFAC. To interpret the decomposition, PARAFAC relies on some strong _chemical_ assumptions (not just statistical), namely:

1. There are no inner filter effects occurring
2. No quenching is present
3. Beer-Lambert law is satisfied
4. No additional scattering is present


If we generalize to non-linear decomposition, and ignore any attempt at interpretation, we can expand the models used. Namely, we can try a convolutional autoencoder to project the 2D EEMs to a lower space, and perform analysis there.



### Installation

1. Clone/download the repo to a local directory.
2. Optional: create a virtualenv for this.
3. From the command line:
```
python setup.py
```

### Configuration

1. Currently the supported EEMs must be NxN (a square). One can use image / scientific software to resize EEMs to be square. Change the `INPUTS` variable in `src/utils`.
2. Data, in the form of csv (with `.csv` extension), should be put into the folder `data/flat_files`.


### Running on an example dataset

```
python src/keras_conv_ae_training.py && python src/keras_encoder_prediction.py
```
