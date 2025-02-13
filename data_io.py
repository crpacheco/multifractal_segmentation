import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rich.table import Table

def save_true_image(arr, filename):
  plt.imshow(arr, cmap = 'tab10')
  plt.axis("off")   # turns off axes
  plt.axis("tight")  # gets rid of white border
  plt.axis("image")  # square up the image instead of filling the "figure" space
  plt.savefig(filename, bbox_inches='tight', pad_inches=0)
  plt.close()

def save_image(arr, filename, use_mean=True, use_std=True):
  if use_mean and use_std:
    plt.imshow(arr, vmax = np.mean(arr) + np.std(arr), cmap = 'gray')
  elif use_mean and not use_std:
    plt.imshow(arr, vmax = np.mean(arr), cmap = 'gray')
  else:
    plt.imshow(arr, cmap = 'gray')
  plt.axis("off")   # turns off axes
  plt.axis("tight")  # gets rid of white border
  plt.axis("image")  # square up the image instead of filling the "figure" space
  plt.savefig(filename, bbox_inches='tight', pad_inches=0)
  plt.close()

def save_raster(arr, filename, template_filename):
  with rasterio.open(template_filename) as src:
    meta = src.meta
    meta.update(count = 1)
    with rasterio.open(filename, 'w', **meta) as dst:
      dst.write(arr, 1)

def get_user_labels(user_labeled_zones_path, mfs_img_shape):
  with rasterio.open(user_labeled_zones_path, 'r') as src:
    label_mask = src.read(1)

    nr, nc = mfs_img_shape
    label_mask_flatten = label_mask[0:nr, 0:nc].flatten()
    pos = np.nonzero(label_mask_flatten)
    targets = label_mask_flatten[pos]

    return (pos, targets)

def array_to_table(arr):
  table = Table(show_header = False, show_lines=True)
  r, c = arr.shape
  for j in range(c):
    table.add_column('', justify='center')

  conv = lambda v: "{:.2f}".format(v)
  for i in range(r):
    table.add_row(*list(map(conv, arr[i])))

  return table