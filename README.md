## Autoencoding EEMs


EEMs (excitation emission matrices) are measurements of a sample's fluorescence intensity at varying excitation and emission wavelengths.



### Analysis of EEMs

Traditionally, EEMs have been analyzed using linear matrix decomposition methods like PARAFAC. To interpret the decomposition, PARAFAC relies on some strong _chemical_ assumptions (not just statistical), namely:

1. There are no inner filter effects occurring
2. No quenching is present
3. Beer-Lambert law is satisfied
4. No additional scattering is present


If we generalize to non-linear decomposition, and ignore any attempt at interpretation, we can expand the models used. Namely, we can try a convolutional autoencoder.



### Installation

Clone/download the repo to a local directory. From the command line:

```
python setup.py
```


### Running on an example dataset
