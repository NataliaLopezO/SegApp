import numpy as np
from scipy.signal import find_peaks
import scipy.stats as stats
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



def white_stripe(X):

    # Calcula el histograma
    hist, bins = np.histogram(X.ravel(), bins="auto")

    # Encuentra los picos del histograma
    peaks, _ = find_peaks(hist)

    # Si hay al menos tres picos, utiliza el valor moda entre el segundo y el tercer pico como divisor
    if len(peaks) >= 3:
        last_peak = peaks[-1]
        #second_last_peak = peaks[-2]
        start_index = max(0, last_peak - 10)
        last_peak_range = range(int(bins[start_index]), int(bins[-1]) + 1)
        #second_last_peak_range = range(int(bins[second_last_peak]), int(bins[last_peak])+1)
        mode, _ = stats.mode(hist[last_peak_range])
        divisor = mode[0]
    # Si hay menos de tres picos, utiliza el valor moda de todo el histograma como divisor
    else:
        mode, _ = stats.mode(hist)
        divisor = mode[0]

    # Divide el histograma por el valor divisor
  #hist_norm = hist / divisor
    image_ws = X / divisor

    return image_ws

def histogram_matching(image_data):
    ## Load the original image data
    data_orig = image_data
    # Load the target image data
    #path = os.path.abspath("./images/1/FLAIR.nii.gz")
    data_target = nib.load('uploaded_images/T1.nii.gz').get_fdata()

    # Flatten the data arrays into 1D arrays
    flat_orig = data_orig.flatten()
    flat_target = data_target.flatten()

    # Calculate the cumulative histograms for the original and target images
    hist_orig, bins = np.histogram(flat_orig, bins=256, range=(0, 255), density=True)
    hist_orig_cumulative = hist_orig.cumsum()
    hist_target, _ = np.histogram(flat_target, bins=256, range=(0, 255), density=True)
    hist_target_cumulative = hist_target.cumsum()

    # Map the values of the original image to the values of the target image
    lut = np.interp(hist_orig_cumulative, hist_target_cumulative, bins[:-1])

    # Apply the mapping to the original image data
    data_matched = np.interp(data_orig, bins[:-1], lut)

    return data_matched

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
