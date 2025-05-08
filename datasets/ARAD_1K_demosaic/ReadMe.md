The original ARAD_1K dataset, which contains Hyperspectral Images (HSI) and ideal RGB images, can be downloaded from [this link](https://codalab.lisn.upsaclay.fr/competitions/721#participate-get-data).

To make the data more realistic, we generate demosaiced RGB images. You can run the `generate_rgb.py` script for testing purposes.

The mask used is a center - cropped version of the real mask from [TSA-Net](https://github.com/mengziyi64/TSA-Net.git).