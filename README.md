# inpainting

## Prerequisites

## Download data
1) download data in `./data` folder
- for the complete set of data: [link to base-chl.tgz](https://mycore.core-cloud.net/index.php/s/XwQZHm37ziEFtPM)
- for a small set (for quick testing): [link to base-chl-small.tgz](https://mycore.core-cloud.net/index.php/s/90Lr8u83YP8pDzT)
2) Uncompress the data with `tar -xvzf base-chl.tgz` or `tar -xvzf base-chl-small.tgz`

## Create training set
1) In file `make-trainset.py`: 
- set the `basename` to `medchl.nc` or `medchl-small.nc`
- set the `trainingname` denoted `[TNAME]`
- check the options (e.g. king of masking)
2) run the file in a terminal: `./make-trainset.py`
3) Check some randomly picked images in `./figures/examples/[TNAME]`

## Train a first network
1) In file `Inpainting.py`:
- set the `trainingname` to `[TNAME]`
- set the `name` to the name the neural network has to be saved (denoted `[NETNAME]`)
2) run the file in a terminal: `./Inpainting.py`

## Test the model
1) In file `test_model.py`:
- set the `name` to `[NETNAME]`
2) run the file in a terminal : `./test_model.py`
3) Check some randomly picked images in `./figures/exemples/[NETNAME]`

