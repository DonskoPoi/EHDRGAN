# EHDRGAN
An enhanced GAN network for single photo HDR

# Requirements

- Please check requirements.txt
- Use `pip install requirements.txt` for quick install

# Training

### Dataset
1. Prepare your dataset in [NTIRE2021 HDR Challenge](https://competitions.codalab.org/competitions/28161#participate-get-data), please register an account in the challenge homepage
2. The dataset file is restored in `EHDRGAN/data`, you can modify your own dataset in that folder with a file name like`xxx_dataset.py` and adding a decorator `@DATASET_REGISTRY.register()`
### Train Pipline
1. Modify [train config file](options/train_hdrunet.yml)
2. Usage: `python utils/train.py -opt <yml path>`
3. Please check [basicSR](https://github.com/XPixelGroup/BasicSR) for train pipline
### Pretrained Weights
A pretrained weight for HDRUNet is restored in `HDRGAN/weights`, you can either use it for pretraining or testing

# Testing
1. Modify [test config file](options/test.yml)
2. Generate HDR images: `python utils/inferece_img.py -opt <yml path>`
3. Check [testing file](utils/test.py)
4. Get result: `python utils/test.py` and the result will automatically appear in yor console with log form
