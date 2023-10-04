import nibabel as nib
import pyrobex.robex as rob
import numpy as np

def mask(ruta_IR, path_stripped, path_mask):
  data = nib.load(ruta_IR)
  stripped, mask = rob.robex(data)
  ##para guardar la mascara
  nib.save(stripped, path_stripped)
  nib.save(mask, path_mask)
  stripped_data = nib.load(path_stripped).get_fdata()
  mask_data = nib.load(path_mask).get_fdata()

  return mask_data

def no_skull(mask, image):  

  depth, height, width = image.shape
  filtered_image = np.zeros_like(image)
  for x in range(1, depth - 1):
      for y in range(1, height - 1):
          for z in range(1, width - 1):
            if np.abs(mask[x,y,z]) == 0:
              filtered_image[x, y, z] = 0
            else:
              filtered_image[x, y, z] = image[x,y,z]

  return filtered_image