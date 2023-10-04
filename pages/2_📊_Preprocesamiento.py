import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import io

from algoritmos.estandarizacion import rescaling, zscore, white_stripe,histogram_matching, mean_filter_3d, median_filter_3d, meanwithBorder

st.set_page_config(page_title="Estandarización & eliminación de ruido", page_icon="📊")


st.markdown("# Estandarización & eliminación de ruido")
st.sidebar.header("Preprocesamiento")
st.write(
    """Selecciona una imagen para comenzar a procesarla"""
)


nombreImagenes = []

folder_path = "uploaded_images"  # Ruta de la carpeta que deseas listar
# Verifica si la carpeta existe
if os.path.exists(folder_path):
    # Enumera los archivos en la carpeta
    files = os.listdir(folder_path)
    
    # Itera sobre cada archivo
    for filename in files:
        # Muestra el nombre del archivo
        nombreImagenes.append(filename)
else:
    st.error(f"La carpeta '{folder_path}' no existe")



folder_path_seg = "seg_images" 
if st.sidebar.button("Borrar segmentaciones anteriores"):
    # Verifica si la carpeta existe
    if os.path.exists(folder_path_seg):
        # Enumera los archivos en la carpeta
        files = os.listdir(folder_path_seg)

        # Itera sobre cada archivo y lo borra
        for filename in files:
            file_path = os.path.join(folder_path_seg, filename)
            os.remove(file_path)
        
        st.success("Archivos borrados correctamente")
        st.experimental_rerun()
    else:
        st.error(f"La carpeta '{folder_path_seg}' no existe")


imagen = st.sidebar.selectbox('Selecciona una opción', nombreImagenes, index=nombreImagenes.index('T1.nii.gz'))

opciones_algoritmos = ['Ninguno','Rescaling', 'Z-Score', 'White Stripe','Histogram Matching']
algoritmo = st.sidebar.radio('Selecciona un algortimo de estandarización', opciones_algoritmos)

opciones_ruido = ['Ninguno','Mean Filter', 'Median Filter', 'Median Filter with edges']
ruido = st.sidebar.radio('(Opcional) Selecciona un algortimo de eliminación de ruido', opciones_ruido)

# Dividir el espacio en dos columnas
col1, col2 = st.columns(2)


path = folder_path+"/"+imagen
image_load = nib.load(folder_path+"/"+imagen)
image_data = image_load.get_fdata()  

if 'imagen_preprocesada' not in st.session_state:
    st.session_state.imagen_preprocesada = image_data

if 'name_imagen_preprocesada' not in st.session_state:
    st.session_state.name_imagen_preprocesada = imagen

if 'imagen_datos' not in st.session_state:
    st.session_state.imagen_datos = image_load



#####cargar histograma 

if algoritmo != 'Ninguno':
    with col1:
        hist1_data = image_data[image_data > 10].flatten()
        fig1, ax1 = plt.subplots()
        ax1.hist(hist1_data, bins=100)
        st.write("Histograma sin estandarización")
        st.pyplot(fig1)        


    with col2:
        if algoritmo == 'Rescaling':
            image_data_rescaled = rescaling(image_data)
            hist2_data = image_data_rescaled[image_data_rescaled > 0.01].flatten()
            fig2, ax2 = plt.subplots()
            ax2.hist(hist2_data, bins=100)
            
        if algoritmo == 'Z-Score':
            image_data_rescaled = zscore(image_data)
            hist2_data = image_data_rescaled.flatten()
            fig2, ax2 = plt.subplots()
            ax2.hist(hist2_data, bins=100)

        if algoritmo == 'White Stripe':
            image_data_rescaled = white_stripe(image_data, imagen)
            hist2_data = image_data_rescaled[image_data_rescaled>0.5 ].flatten()
            fig2, ax2 = plt.subplots()
            ax2.hist(hist2_data, bins=100)

        if algoritmo == 'Histogram Matching':
            ks = st.number_input("Percentiles:", value=3)
            image_data_rescaled = histogram_matching(image_data, ks, imagen)
            hist2_data = image_data_rescaled[image_data_rescaled>10].flatten()
            fig2, ax2 = plt.subplots()
            ax2.hist(hist2_data, bins=100)

        st.write("Histograma con estandarización")
        st.pyplot(fig2)
        
    st.session_state.imagen_preprocesada =  image_data_rescaled
    st.session_state.name_imagen_preprocesada = imagen
    st.session_state.imagen_datos = image_load


if ruido != 'Ninguno':
    with col1:
 
        fig, ax = plt.subplots(figsize=(50, 50))

        if(imagen == 'T1.nii.gz'):
            opciones = [ 'Axial', 'Sagital', 'Coronal']
            corte = st.selectbox('1.Selecciona un corte', opciones)

            if corte == 'Axial':
                valor_especifico = 100
                valor_maximo = image_data_rescaled.shape[1]
                ####slider
                valor_seleccionado = st.slider("1.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(image_data_rescaled[:, valor_seleccionado, :], cmap='bone')

            if corte == 'Sagital':
                valor_especifico = 100
                valor_maximo = image_data_rescaled.shape[2]
                ####slider
                valor_seleccionado = st.slider("1.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(np.rot90(image_data_rescaled[:, :, valor_seleccionado],k=-1), cmap='bone')

            if corte == 'Coronal':
                valor_especifico = 100
                valor_maximo = image_data_rescaled.shape[0]
                ####slider
                valor_seleccionado = st.slider("1.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(image_data_rescaled[valor_seleccionado, :, :], cmap='bone')


        if(imagen == 'IR.nii.gz' or imagen == 'FLAIR.nii.gz'):
            valor_especifico = 25
            valor_maximo = image_data_rescaled.shape[2]
            ###imagen
            valor_seleccionado = st.slider("Imagen sin eliminación de ruido", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(np.rot90(image_data_rescaled[:, :, valor_seleccionado], k=-1), cmap='bone')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Mostrar la imagen utilizando st.image
        st.image(buffer, caption = imagen, use_column_width=True)

    with col2:
  
        fig, ax = plt.subplots(figsize=(50, 50))

        if ruido == 'Mean Filter':
            filtered_image = mean_filter_3d(image_data_rescaled)
        
        if ruido == 'Median Filter':
            filtered_image = median_filter_3d(image_data_rescaled)

        if ruido == 'Median Filter with edges':
            filtered_image = meanwithBorder(image_data_rescaled)

        if(imagen == 'T1.nii.gz'):
            opciones = ['Axial', 'Sagital', 'Coronal']
            corte = st.selectbox('2.Selecciona un corte', opciones)

            if corte == 'Axial':
                valor_especifico = 100
                valor_maximo = filtered_image.shape[1]
                ####slider
                valor_seleccionado = st.slider("2.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(filtered_image[:, valor_seleccionado, :], cmap='bone')

            if corte == 'Sagital':
                valor_especifico = 100
                valor_maximo = filtered_image.shape[2]
                ####slider
                valor_seleccionado = st.slider("2.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(np.rot90(filtered_image[:, :, valor_seleccionado],k=-1), cmap='bone')

            if corte == 'Coronal':
                valor_especifico = 100
                valor_maximo = filtered_image.shape[0]
                ####slider
                valor_seleccionado = st.slider("2.Selecciona un valor", 0, valor_maximo, valor_especifico)
                ###imagen
                ax.imshow(filtered_image[valor_seleccionado, :, :], cmap='bone')
        if(imagen == 'IR.nii.gz' or imagen == 'FLAIR.nii.gz'):
            valor_especifico = 25
            valor_maximo = filtered_image.shape[2]
            ###imagen
            valor_seleccionado = st.slider("Imagen con eliminación de ruido", 0, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(np.rot90(filtered_image[:, :, valor_seleccionado], k=-1), cmap='bone')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Mostrar la imagen utilizando st.image
        st.image(buffer, caption = imagen, use_column_width=True)
        

    st.session_state.imagen_preprocesada =  filtered_image
    st.session_state.name_imagen_preprocesada = imagen
    st.session_state.imagen_datos = image_load
    