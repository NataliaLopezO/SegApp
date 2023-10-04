import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import io 
import os
import pandas as pd

from algoritmos.registro import registrar
from algoritmos.no_skull import mask, no_skull
from algoritmos.volumenes import calculate_cluster_volumes


st.set_page_config(page_title="Registro", page_icon="📊")

def descargar_imagen():
    with open('reg_images/register.nii.gz', "rb") as archivo:
        contenido = archivo.read()
    return contenido


st.markdown("# Registro")
st.sidebar.header("Registro")
st.sidebar.write(
    """Selecciona una imagen fixed y una moving para el registro"""
)

#name_imagen = st.session_state.name_imagen_preprocesada
#image_load = st.session_state.imagen_datos


clusters = []


nombreImagenes = []

##Cargar imagenes sin segmentar 
folder_path = "uploaded_images"
if os.path.exists(folder_path):
    files = os.listdir(folder_path)
    for filename in files:
        nombreImagenes.append(folder_path+"/"+filename)
else:
    st.error(f"La carpeta '{folder_path}' no existe")

imagenesSeg = []
##cargar imagenes segmentadas 
folder_path_seg = "seg_images" 
if os.path.exists(folder_path_seg):
    files = os.listdir(folder_path_seg)
    for filename in files:
        nombreImagenes.append(folder_path_seg+"/"+filename)
else:
    st.error(f"La carpeta '{folder_path_seg}' no existe")

fixed_imagen_trans = st.sidebar.selectbox('Selecciona la imagen fixed para la transformación', nombreImagenes, index=nombreImagenes.index('uploaded_images/FLAIR.nii.gz'))
moving_imagen_trans = st.sidebar.selectbox('Selecciona la imagen moving para la transformación', nombreImagenes, index=nombreImagenes.index('uploaded_images/T1.nii.gz'))

fixed_imagen_reg = st.sidebar.selectbox('Selecciona la imagen fixed para el registro', nombreImagenes+imagenesSeg, index=nombreImagenes.index('uploaded_images/FLAIR.nii.gz'))
moving_imagen_reg  = st.sidebar.selectbox('Selecciona la imagen moving para el registro', nombreImagenes+imagenesSeg, index=nombreImagenes.index('seg_images/seg_T1.nii.gz'))

url = registrar(fixed_imagen_trans, moving_imagen_trans, fixed_imagen_reg, moving_imagen_reg, 'reg_images/register.nii.gz')
    
registered_nifti = nib.load(url)
registered_data = registered_nifti.get_fdata()

   
button_skull = st.button("Remover craneo")
if button_skull:
    mask_segmentation = mask("uploaded_images/IR.nii.gz", "skull_images/stripped.nii.gz",  "skull_images/mask.nii.gz")
    no_skull_re = no_skull(mask_segmentation , registered_data)
    image_FLAIR = nib.load('uploaded_images/FLAIR.nii.gz')
    nifti_image_skull = nib.Nifti1Image(no_skull_re, image_FLAIR.affine, image_FLAIR.header)
    nifti_image_skull.to_filename("reg_images/register.nii.gz")

registered_nifti = nib.load(url)
registered_data = registered_nifti.get_fdata()
valor_especifico = 25
valor_maximo = registered_data.shape[2]
valor_seleccionado = st.slider("Selecciona una coordenada", 0, valor_maximo, valor_especifico)
fig, ax = plt.subplots()
ax.imshow(np.rot90(registered_data[:, :, valor_seleccionado],k=-1), cmap='bone')
fig.set_size_inches(2, 2) 
st.write(
    """## Resultado del registro"""
    )
st.pyplot(fig)

##Tabla con los volumenes
dic_clusters = calculate_cluster_volumes('reg_images/register.nii.gz')
for key in dic_clusters.keys():
    clusters.append(dic_clusters[key])
st.write("""## Volumenes de los clusters""")
data = {
    'Volumen del cluster (mm3)': clusters
}
df = pd.DataFrame(data)
st.table(df)

st.write("""## Descargar imagen""")
st.download_button(
    label="Descargar imagen",
    data=descargar_imagen(),
    file_name="register.nii.gz",

)




