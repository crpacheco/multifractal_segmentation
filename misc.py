import numpy as np

def normalize_data(data, vmin=0, vmax=1):
  mn = np.min(data)
  sign = -1 if mn >= 0 else 1
  diff = abs(mn - vmin) * sign
  ndata = data + diff
  mx = np.max(ndata)
  return ndata / mx

def majority(w):
  values, counts = np.unique(w, return_counts=True)
  pos = np.argmax(counts)
  return values[pos]