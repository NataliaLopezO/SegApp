import numpy as np
import nibabel as nib

def calculate_cluster_volumes(segmentation):

    image_header = nib.load(segmentation).header
    pixdim = image_header['pixdim']  # Obtener el atributo 'pixdim' del encabezado
    pixel_size = np.prod(pixdim[1:4])

    image_data = nib.load(segmentation).get_fdata()
    #unique_labels = np.unique(image_data)
    unique_labels = np.unique(image_data.astype(int))


    cluster_volumes = {}
    for label in unique_labels:


        cluster_mask = (image_data == label)  # Crear una máscara para el cluster actual
        cluster_pixels = np.sum(cluster_mask)
        cluster_volume = cluster_pixels * pixel_size

        cluster_volumes[label] = cluster_volume

    return cluster_volumes