# SeisDPS-3D
this is the code implementation of "[SeisDPS-3D: A Diffusion Model-Based Approach for 3D Seismic Data Deconvolution]

This place only includes testing for 2D data. The 3D data testing code will be added in a future update.

# Train

## Training models

### Preparing Data

The training code reads npy data from a directory of data files. 

For creating your own dataset, simply dump all of your datas into a directory with ".npy" extensions. 

## training

```
python train.py 
```
## Reconstruction
The SeisDPS.yml contains information about your diffusion model and specifies the location of the weights file.

Download the weight file [here](https://drive.google.com/file/d/1xtQz_51StzLZBtRZyurl_pmGIw5D7EwE/view?usp=drive_link) .

```
python test.py
```
