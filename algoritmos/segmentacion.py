import numpy as np

def isoData(image_data, tau, tol):
  #El tau es un valor entre el minimo y el maximo de la imagen
  
  while True:
    #umbralizar
    segmentation = image_data >= tau

    #calcular un umbral mas optimo
    ##excluyendo los valores del fondo que son muy pequeños
    mBG = image_data[segmentation == 0 ].mean()
    mFG = image_data[segmentation == 1 ].mean()

    tau_post = 0.5 *(mBG + mFG)

    if np.abs(tau-tau_post)<tol:
      break
    else:
      tau = tau_post

  return segmentation


def region_growing(image, x, y, z, tol):
    segmentation = np.zeros_like(image)
    if segmentation[x,y,z] == 1:
        return
    valor_medio_cluster = image[x,y,z]
    segmentation[x,y,z] = 1
    vecinos = [(x, y, z)]
    while vecinos:
        x, y, z = vecinos.pop()
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    #vecino
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if nx >= 0 and nx < image.shape[0] and \
                        ny >= 0 and ny < image.shape[1] and \
                        nz >= 0 and nz < image.shape[2]:
                        if np.abs(valor_medio_cluster - image[nx,ny,nz]) < tol and \
                            segmentation[nx,ny,nz] == 0:
                            segmentation[nx,ny,nz] = 1
                            vecinos.append((nx, ny, nz))
    return segmentation

def clustering(image, ks):

  k_values = np.linspace(np.amin(image), np.amax(image), ks)

  old_segmentation = None

  while True:
    d_values = [np.abs(k-image)for k in k_values]

    segmentation = np.argmin(d_values, axis=0)

    for k_id in range(ks):
      k_values[k_id] = np.mean(image[segmentation==k_id])

    if old_segmentation is not None and np.array_equal(segmentation, old_segmentation):
      break

    old_segmentation = segmentation

  return segmentation

def gmm(image, ks):

  mu_values = np.linspace(np.amin(image), np.amax(image), ks)
  sd_values = np.ones_like(mu_values)

  old_segmentation = None

  while True:
    d_values = [np.exp(-0.5 * np.divide((image - mu_values[i])**2, np.maximum(sd_values[i]**2, 1e-10))) for i in range(ks)]

    segmentation = np.argmax(d_values, axis=0)
    for i in range(ks):
      mu_values[i] = np.mean(image[segmentation==i])
      

    if old_segmentation is not None and np.array_equal(segmentation, old_segmentation):
      if len(np.unique(segmentation)) == ks:
                break

    old_segmentation = segmentation

  return segmentation























#sd_values[i] = np.std(image[segmentation==i])