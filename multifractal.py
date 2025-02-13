import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import misc

from math import exp, log10, ceil

def get_kernel(size, sigma = 2):
  """ Returns a normalized 2D gauss kernel array for convolutions """
  m = np.float32(size)
  n = np.float32(size)
  if(size <= 3): sigma = 1.5;
  if(size == 5): sigma = 2.5;
  y, x = np.mgrid[-(m-1)/2:(m-1)/2+1, -(n-1)/2:(n-1)/2+1]

  b = 2*(sigma**2)
  x2 = list(map(lambda i: list(map( lambda j: j**2,i)), x))
  y2 = list(map(lambda i: list(map( lambda j: j**2,i)), y))
  g = np.sum([x2,y2],axis=0).astype(np.float32)
  g = np.array(list(map(lambda i: list(map( lambda j: exp(-j/b),i)), g))).astype(np.float32)
  return g / g.sum()

def calculate_local_density_image(im, averaging):
  #Using [0..255] to denote the intensity profile of the image
  grayscale_box =[0, 255];

  im = im * grayscale_box[1]

  ### Estimating density function of the image
  ### by solving least squares for D in  the equation
  ### log10(bw) = D*log10(c) + b
  r = 1.0/max(im.shape)
  c = np.dot(range(1,averaging+1),r)

  c = np.array(list(map(lambda i: log10(i), c)))
  bw = np.zeros((averaging,im.shape[0],im.shape[1])).astype(np.float32)

  bw[0] = im + 1

  k = 1
  if(averaging > 1):
      bw[1] = scipy.signal.convolve2d(bw[0], get_kernel(k+1),mode="full")[1:,1:]*((k+1)**2)

  for k in range(2,averaging):
      temp = scipy.signal.convolve2d(bw[0], get_kernel(k+1),mode="full")*((k+1)**2)
      if(k==4):
          bw[k] = temp[k-1-1:temp.shape[0]-(k/2),k-1-1:temp.shape[1]-(k/2)]
      else:
          bw[k] = temp[k-1:temp.shape[0]-(1),k-1:temp.shape[1]-(1)]


  bw = np.log10(bw)
  n1 = c[0]*c[0]
  n2 = bw[0]*c[0]

  for k in range(1,averaging):
      n1 = n1+c[k]*c[k]
      n2 = n2 + bw[k]*c[k]

  sum3 = bw[0]
  for i in range(1,averaging):
      sum3 = sum3 + bw[i]

  if(averaging >1):
      D = (n2*averaging-sum(c)*sum3)/(n1*averaging -sum(c)*sum(c));

  if (averaging > 1):
      max_D  = np.float32(4)
      min_D = np.float32(1)
      D = grayscale_box[1]*(D-min_D)/(max_D - min_D)+grayscale_box[0]
  else:
      D = im

  return D

def calculate_box_counting(levelset_window, iterations, spectrum_size):
  IM = np.zeros(levelset_window.shape)
  r = max(IM.shape)
  c = np.zeros(iterations)
  c[0] = 1;
  for k in range(1,iterations):
      c[k] = c[k-1]/(k+1)
  c = c / sum(c)

  #Estimate MFS by box-counting
  num = np.zeros(iterations)
  MFS = np.zeros(spectrum_size)
  for k in range(1,spectrum_size+1):
      IM = np.zeros(IM.shape)
      IM = (levelset_window==k).choose(levelset_window,255+k)
      IM = (IM<255+k).choose(IM,0)
      IM = (IM>0).choose(IM,1)
      temp = max(IM.sum(),1)
      num[0] = log10(temp)/log10(r)
      for j in range(2,iterations+1):
          mask = np.ones((j,j))
          bw = scipy.signal.convolve2d(IM, mask,mode="full")[1:,1:]
          indx = np.arange(0,IM.shape[0],j)
          indy = np.arange(0,IM.shape[1],j)
          bw = bw[np.ix_(indx,indy)]
          idx = (bw>0).sum()
          temp = max(idx,1)
          num[j-1] = log10(temp)/log10(r/j)

      MFS[k-1] = sum(c*num)

  return MFS


def calculate_multifractal_spectrum(img, spectrum_size, averaging, iterations, sliding_window_size=None):
  density_image = calculate_local_density_image(img, averaging)

  #Partition the density
  # throw away the boundary
  D = density_image[averaging-1:density_image.shape[0]-averaging+1, averaging-1:density_image.shape[1]-averaging+1]
  D = misc.normalize_data(D)


  center = np.linspace(np.min(D),np.max(D), spectrum_size)
  gap = center[1] - center[0]
  IM = (D  * 200/ gap).astype(int) + 1

  # calculate the fractal dimensions for each window instance
  if not sliding_window_size:
    sliding_window_size = IM.shape
  windows = np.lib.stride_tricks.sliding_window_view(IM, sliding_window_size) # sliding window (array of matrices of matrices)
  nr, nc, wr, wc = windows.shape
  pixels_count = nr * nc
  windows = windows.reshape((pixels_count, wr, wc)) # sliding window (array of matrices)

  spectra = np.zeros((pixels_count, spectrum_size))

  for j in range(pixels_count):
      spectra[j] = calculate_box_counting(windows[j], iterations, spectrum_size)
  return (center, spectra, (nr, nc))
