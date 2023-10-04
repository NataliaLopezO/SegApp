import numpy as np
from scipy.signal import find_peaks
import nibabel as nib

def rescaling(image):
  min_value = image.min()
  max_value = image.max()

  image_data_rescaled = (image - min_value) / (max_value - min_value)
  return image_data_rescaled


def zscore(image):
  mean = image[image > 10].mean()
  standard_deviation = image[image > 10].std()
  image_zscore = (image - mean)/(standard_deviation) 
  return image_zscore



def white_stripe(X, tipo):
  hist, big_edges = np.histogram(X.flatten(), bins= 'auto')
  picos, _ = find_peaks(hist, height= 100)
  val_picos = big_edges[picos]

  if tipo == 'FLAIR.nii.gz' or tipo == 'T1.nii.gz':
    image_data_rescaled = X/ val_picos[1]

  if tipo == 'IR.nii.gz':
    image_data_rescaled = X/val_picos[0]

  return image_data_rescaled


def histogram_matching(image_data, ks, ref):

  if ref == 'T1.nii.gz':
    data_target = nib.load('ref_images/T1_copia.nii.gz').get_fdata()
  if ref == 'IR.nii.gz':
    data_target = nib.load('ref_images/IR_copia.nii.gz').get_fdata()
  if ref == 'FLAIR.nii.gz':
    data_target = nib.load('ref_images/FLAIR_copia.nii.gz').get_fdata()   

  ini=0
  fin = 100
  step = (fin - ini)/(ks-1)

  percentiles_data = np.arange(ini, fin+step, step)
  percentiles_target = np.arange(ini, fin+step, step)

  p1 = np.percentile(image_data, percentiles_data)
  p2 = np.percentile(data_target, percentiles_target)

  
  return np.interp(image_data, p1, p2)


def mean_filter_3d(image):
    depth, height, width = image.shape
    filtered_image = np.zeros_like(image)
    for z in range(1, depth - 1):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                neighbors = [
                    image[z-1, y, x],
                    image[z+1, y, x],
                    image[z, y-1, x],
                    image[z, y+1, x],
                    image[z, y, x-1],
                    image[z, y, x+1],
                    image[z, y, x]
                ]
                filtered_value = np.mean(neighbors)
                filtered_image[z, y, x] = filtered_value

    return filtered_image

def median_filter_3d(image):
    depth, height, width = image.shape
    filtered_image = np.zeros_like(image)
    for z in range(1, depth - 1):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                neighbors = [
                    image[z-1, y, x],
                    image[z+1, y, x],
                    image[z, y-1, x],
                    image[z, y+1, x],
                    image[z, y, x-1],
                    image[z, y, x+1],
                    image[z, y, x]
                ]
                filtered_value = np.median(neighbors)
                filtered_image[z, y, x] = filtered_value

    return filtered_image

def derivada(image):
  df = np.zeros_like(image)
  for x in range(1, image.shape[0] - 2):
    for y in range(1, image.shape[1] - 2):
      for z in range(1, image.shape[2] - 2):
        dfdx = np.power( image[x+1, y, z]- image[x-1,y,z] , 2)
        dfdy = np.power( image[x, y+1, z]- image[x,y-1,z] , 2)
        dfdz = np.power( image[x, y, z+1]- image[x,y,z-1] , 2)
        df[x,y,z] =np.sqrt(dfdx + dfdy + dfdz)
  return df


def meanwithBorder(image_data, tol=10):
  df = derivada(image_data)
  depth, height, width = image_data.shape
  filtered_image = np.zeros_like(image_data)
  for x in range(1, depth - 1):
      for y in range(1, height - 1):
          for z in range(1, width - 1):
              if np.abs(df[x,y,z]) < tol:
                neighborhood = image_data[x-1:x+2 , y- 1: y+2 , z-1: z+2]
                filtered_value = np.mean(neighborhood)
                filtered_image[x, y, z] = filtered_value

              else:
                filtered_image[x, y, z] = image_data[x, y, z]

  return filtered_image