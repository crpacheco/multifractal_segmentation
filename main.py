import argparse
import os

import rasterio
import numpy as np

import data_io
import multifractal as mfr
import data_io as dio
import misc
import validation

from sklearn import preprocessing as pre
from minisom import MiniSom
from scipy.ndimage import generic_filter
from rich import print

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('sar_image', help='Input SAR data filename', type=str)
    parser.add_argument('user_label_mask', help='Input user labeled pixeles filename', type=str)
    parser.add_argument('output_directory', help='Output directory', type=str)
    parser.add_argument('--mfs-pixel_averaging-count', default=1, type=int, help='Number of pixels to average in sliding window to calculate MFS pixels')
    parser.add_argument('--mfs-spectrum-size', default=3, type=int, help='Length of multifractal spectrum to calculate in each instance of the sliding window')
    parser.add_argument('--mfs-iterations', default=3, type=int, help='Number of iterations to calculate multifractal spectrum')
    parser.add_argument('--mfs-window-size', help='Window size for multifractal spectrum', type=int, default=10)
    args = parser.parse_args()

    # MFS parameters
    pixel_averaging_count = args.mfs_pixel_averaging_count
    spectrum_size = args.mfs_spectrum_size
    iterations = args.mfs_iterations
    win_shape = (args.mfs_window_size, args.mfs_window_size)

    # image parameters
    img_path = args.sar_image
    user_labeled_zones_path = args.user_label_mask
    output_directory = args.output_directory

    # read the actual sar data content
    print('[bold italic blue]Reading SAR data...')
    with rasterio.open(img_path, 'r') as src:
        img = src.read(1)
        meta = src.meta
        tags = src.tags()

        validate = (src.count > 1)
        if validate:
            img_gt = src.read(2)

    if img_path.endswith('subset_monte_hermoso.tif'):
        img_gt[np.nonzero(img_gt == 3)] = 4  # replace four pixels with wrong label

    # normalized images generation
    print('[bold italic blue]Using transforms on SAR data...')
    fimg = img.flatten().reshape(-1, 1)
    nimg = pre.MinMaxScaler().fit_transform(pre.QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='uniform').fit_transform(fimg)).reshape(img.shape)
    nimg2 = pre.MinMaxScaler().fit_transform(pre.PowerTransformer().fit_transform(fimg)).reshape(img.shape)

    # MFS matrices calculation
    print('[bold italic blue]Calculating the multifractal spectrum...')
    alphas, features_matrix, mfs_img_shape = mfr.calculate_multifractal_spectrum(nimg, spectrum_size, pixel_averaging_count, iterations, win_shape)
    alphas2, features_matrix2, _ = mfr.calculate_multifractal_spectrum(nimg2, spectrum_size, pixel_averaging_count, iterations, win_shape)
    nr, nc = mfs_img_shape

    # MFS submatrix generation from user labeled pixels
    print('[bold italic blue]Connecting the labels to the right rows of the MFS matrix....')
    pos, targets = dio.get_user_labels(user_labeled_zones_path, mfs_img_shape)
    subset = features_matrix[pos]

    # SOM training
    print('[bold italic blue]Training AI model...')
    m, n = 10, 10
    dim = subset.shape[1]
    som = MiniSom(m, n, dim, sigma=0.5, learning_rate=0.5, neighborhood_function='gaussian', activation_distance='euclidean', random_seed=10)  # initialization of 6x6 SOM
    som.train_batch(subset, 100000)  # trains the SOM with 100000 iterations

    # BML generation
    print('[bold italic blue]Building BMU-label map...')
    subset_winner_coordinates = np.array([som.winner(x) for x in subset]).T
    tlabels = np.unique(targets).tolist()
    M = np.zeros((len(tlabels), m, n), dtype=int)
    for k in range(subset_winner_coordinates[0].size):
        i = subset_winner_coordinates[0, k]
        j = subset_winner_coordinates[1, k]
        p = tlabels.index(targets[k])
        M[p, i, j] += 1
    n_gt = np.argmax(M, axis=0)

    # segmented image generation
    print('[bold italic blue]Doing segmentation...')
    winner_coordinates = np.array([som.winner(x) for x in features_matrix]).T
    labeled = np.zeros(winner_coordinates[0].size)
    for k in range(winner_coordinates[0].size):
        i = winner_coordinates[0, k]
        j = winner_coordinates[1, k]
        labeled[k] = n_gt[i, j]

    # applying majority filter
    print('[bold italic blue]Applying majority filter...')
    window_size = (15, 15)
    mj_img = generic_filter(labeled.reshape(mfs_img_shape), function=misc.majority, size=window_size)
    data_io.save_raster(mj_img, os.path.join(output_directory, 'mfs_mjc.tif'), img_path)

    if validate:
        print('[bold italic blue]Validating...')
        y_pred = mj_img.flatten()
        y_true = img_gt[0:nr, 0:nc].flatten()
        validation.reclass_data(y_pred, y_true)
        validation.report(y_true, y_pred)

    return 0


if __name__ == '__main__':
    main()