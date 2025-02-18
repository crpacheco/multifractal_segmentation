# SAR multifractal segmentation

This tool facilitates the segmentation and classification of SAR images through the utilisation of multifractal analysis and the employment of self-organising maps.

## Test data 

In the data directory there are three subdirectories:
* **sar_data**: SAR image clippings are found. Subset_dunes.tif is a Sentinel 1 image and subset_monte_hermoso.tif is a SAOCOM image. Both images have a second band with ground truth.
* **synthetic_data**: this directory contains a batch of synthesized images that were generated using the $G^{0}_{I}$ distribution with different values of the alpha and gamma parameters. ![Synthetic SAR image generation parameters](https://github.com/user-attachments/assets/7f2b3f52-cc8d-4e80-b271-a9435a48dc02)
* **user_input**: For each type of SAR images found in the previous directories, there is a mask image in this directory. Each of the mask zones is a subregion of the corresponding SAR region. The supplied masks can be used to test the segmentation script. The user can create his own masks with his own labeled zones.




## Example of use

```
$ python main.py data/synthetic_data/g0_synthetic_0.tif data/user_input/subgt_synthetic.tif <output_directory>
```

In the <output_directory> the segmented image, a validation report and intermediate files are persisted.

```
$ python main.py data/synthetic_data/g0_synthetic_0.tif random <output_directory>
```
If the word **random** is used, the SAR image must have a second band with the ground truth, from which random pixels and their corresponding label will be chosen to perform the model training.
