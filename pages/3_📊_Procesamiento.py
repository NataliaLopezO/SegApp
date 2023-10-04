import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import io 
import os

from algoritmos.segmentacion import isoData, region_growing, clustering, gmm


st.set_page_config(page_title="Segmentación", page_icon="📊")

image_data = st.session_state.imagen_preprocesada
name_imagen = st.session_state.name_imagen_preprocesada
image_load = st.session_state.imagen_datos

seg_path = "seg_images/"


st.markdown("# Segmentación")
st.sidebar.header("Procesamiento")
st.write(
    """Selecciona un algoritmo para segmentar la imagen preprocesada"""
)

opciones_algoritmos = ['Ninguno','isoData', 'Region Growing', 'Clustering','Gaussian Mixture Model']
algoritmo = st.sidebar.selectbox('Selecciona un algortimo de segmentación', opciones_algoritmos)



if algoritmo != 'Ninguno':

    fig, ax = plt.subplots()

    if algoritmo == 'isoData':
        tau = st.sidebar.number_input("Tau:")
        tol = st.sidebar.number_input("Tol:")
        segmentacion = isoData(image_data, tau, tol)
    
    if algoritmo == 'Region Growing':
        tol = st.sidebar.number_input("Tol:", value=142)
        segmentacion = region_growing(image_data, 142,142,142, tol)
    
    if algoritmo == 'Clustering':
        ks = st.sidebar.number_input("Clusters:", value=2)
        segmentacion = clustering(image_data, ks)
    
    if algoritmo == 'Gaussian Mixture Model':
        ks = st.sidebar.number_input("Clusters:", value=2)
        segmentacion = gmm(image_data, ks)

    if name_imagen == 'T1.nii.gz':
        opciones = [ 'Axial', 'Sagital', 'Coronal']
        corte = st.selectbox('Selecciona un corte para ver', opciones)

        if corte == 'Axial':
            valor_especifico = 100
            valor_maximo = segmentacion.shape[1]
            ####slider
            valor_seleccionado = st.slider("Selecciona una coordenada", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(segmentacion[:, valor_seleccionado, :], cmap='bone')

        if corte == 'Sagital':
            valor_especifico = 100
            valor_maximo = segmentacion.shape[2]
            ####slider
            valor_seleccionado = st.slider("Selecciona una coordenada", 0, valor_maximo, valor_especifico)
            ##imagen
            ax.imshow(np.rot90(segmentacion[:, :, valor_seleccionado],k=-1), cmap='bone')

        if corte == 'Coronal':
            valor_especifico = 100
            valor_maximo = segmentacion.shape[0]
            ####slider
            valor_seleccionado = st.slider("Selecciona una coordenada", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(segmentacion[valor_seleccionado, :, :], cmap='bone')


    if(name_imagen == 'IR.nii.gz' or name_imagen == 'FLAIR.nii.gz'):
        valor_especifico = 25
        valor_maximo = segmentacion.shape[2]
        ###imagen
        valor_seleccionado = st.slider("Esta es su imagen segmentada", 0, valor_maximo, valor_especifico)
        ###imagen
        ax.imshow(np.rot90(segmentacion[:, :, valor_seleccionado], k=-1), cmap='bone')

    
    #buffer = io.BytesIO()
    #plt.savefig(buffer, format='png')
    #buffer.seek(0)

    

    # Mostrar la imagen utilizando st.image
    st.write(
        """## Resultado de la segmentación"""
    )
    fig.set_size_inches(2, 2) 
    st.pyplot(fig)
    #st.image(buffer, caption = name_imagen, use_column_width=True)
    
    nifti_image = nib.Nifti1Image(segmentacion, image_load.affine, image_load.header)
    nifti_image.to_filename(seg_path+"seg_"+name_imagen)
    st.write("¡Se ha guardado la imagen!")

# Dividir el espacio en dos columnas
col1, col2 = st.columns(2)



with col1:

    fig, ax = plt.subplots(figsize=(50, 50))

    if(name_imagen == 'T1.nii.gz'):
        opciones = [ 'Axial', 'Sagital', 'Coronal']
        corte = st.selectbox('Selecciona un corte', opciones)

        if corte == 'Axial':
            valor_especifico = 100
            valor_maximo = image_data.shape[1]
            ####slider
            valor_seleccionado = st.slider("Selecciona un valor", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(image_data[:, valor_seleccionado, :], cmap='bone')

        if corte == 'Sagital':
            valor_especifico = 100
            valor_maximo = image_data.shape[2]
            ####slider
            valor_seleccionado = st.slider("Selecciona un valor", 0, valor_maximo, valor_especifico)
            ##imagen
            ax.imshow(np.rot90(image_data[:, :, valor_seleccionado],k=-1), cmap='bone')

        if corte == 'Coronal':
            valor_especifico = 100
            valor_maximo = image_data.shape[0]
            ####slider
            valor_seleccionado = st.slider("Selecciona un valor", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(image_data[valor_seleccionado, :, :], cmap='bone')


    if(name_imagen == 'IR.nii.gz' or name_imagen == 'FLAIR.nii.gz'):
        valor_especifico = 25
        valor_maximo = image_data.shape[2]
        ###imagen
        valor_seleccionado = st.slider("Esta es su imagen", 0, valor_maximo, valor_especifico)
        ###imagen
        ax.imshow(np.rot90(image_data[:, :, valor_seleccionado], k=-1), cmap='bone')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Mostrar la imagen utilizando st.image
    st.image(buffer, caption = name_imagen, use_column_width=True)

with col2:
    st.write(
        """Este es el histograma de su imagen"""
    )
    hist_data3 = image_data.flatten()
    fig3, ax3 = plt.subplots()
    ax3.hist(hist_data3, bins=100)
    st.pyplot(fig3)    



    





