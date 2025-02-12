import numpy as np

def normalize_data(data, vmin=0, vmax=1):
  mn = np.min(data)
  sign = -1 if mn >= 0 else 1
  diff = abs(mn - vmin) * sign
  ndata = data + diff
  mx = np.max(ndata)
  return ndata / mx

def clamp_image(img):
  hcounts, hvalues = np.histogram(img, bins=20)
  pos = np.nonzero((hcounts * 100 / np.sum(hcounts)).astype(int))[0]
  min_value = hvalues[pos[0]]
  max_value = hvalues[pos[-1] + 1]
  x = np.where(img > max_value, max_value, img)
  x = np.where(x < min_value, min_value, x)
  return x

#defining the display function
def GFG(arr,prec):
    np.set_printoptions(suppress=True,precision=prec)
    print(arr)

def majority(w):
  values, counts = np.unique(w, return_counts=True)
  pos = np.argmax(counts)
  return values[pos]