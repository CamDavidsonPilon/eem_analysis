## Autoencoding EEMs


### Analysis of EEMs

EEMs (excitation emission matrices) are measurements of a sample's fluorescence intensity at varying excitation and emission wavelengths.

Traditionally, EEMs have been analyzed using linear matrix decomposition methods like PARAFAC. To interpret the decomposition, PARAFAC relies on some strong _chemical_ assumptions (not just statistical), namely:

1. There are no inner filter effects occurring
2. No quenching is present
3. Beer-Lambert law is satisfied
4. No additional scattering is present


If we generalize to non-linear decomposition, and ignore any attempt at interpretation, we can expand the models used. Namely, we can try a convolutional autoencoder to project the 2D EEMs to a lower space, and perform analysis there. The convolutional autoencoder has a much more accurate compression than alternative methods like PARAFAC. (This also means that the decompression is more accurate, as seen in the image below.)

![comparison](https://i.imgur.com/2t2CdT4.png)


In the comparison above, the convolutional autoencoder, henceforth CNN-AE, squeezes the 28x28 data into 12 dimensions. From these 12 dimensions, further dimensionality reduction can be applied, like PCA. The following figure is a PCA-reduced dataset of four vegetables' EEMS:

![pca](https://i.imgur.com/AwDAdrV.png)

We can clearly see the clusters of vegetables are almost perfectly separated, hence their original EEMs have enough information to distinguish vegetables.


### Existing CNN-AE network

Encoder -> Decoder.


![network](https://i.imgur.com/FRYHunI.png)


### Installation

1. Clone/download the repo to a local directory.
2. Optional: create a virtualenv for this.
3. From the command line:
```
python setup.py
```

### Configuration

1. Currently the supported EEMs must be NxN (a square). One can use image / scientific software to resize EEMs to be square. Change the `INPUTS` variable in `src/utils.py`.
2. Data, in the form of csv (with `.csv` extension), should be put into the folder `data/flat_files`.
3. To added labelling information, you can user `-` delimiters in the filename and edit the `Labels` in `src/utils.py`.


### Running on an example dataset

```
python src/keras_conv_ae_training.py && python src/keras_encoder_reconstruction.py
```
