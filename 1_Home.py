import streamlit as st
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import io

st.set_page_config(
    page_title="SegApp",
    page_icon="🧠",
)

st.write("# Bienvenido a SegApp!")

st.markdown(
    """
    Somos una herramienta enfocada en el procesamiento y procesamiento de imagenes medicas 
    de resonancia magnetica cerebral. La aplicación cuenta con tres modulos importantes, primero 
    está el espacio para el preprocesamiento de la imagen, una sección donde tendremos formas de 
    normalizar las imagenes de entrada para luego trabajar con ellas. Luego, está la sección de 
    procesamiento, para esta parte se le ofrece al usuario diferentes algoritmos de segmentación 
    algunos mas eficientes que otros. Y finalmente, la sección de resultados.

    """
)

st.markdown(
    """
    Te invitamos a cargar la(s) imagen(es) que deseas procesar para iniciar a trabajar con ellas.
    """
)


st.sidebar.success("Select a demo above.")


uploaded_files = st.file_uploader("Choose a .nii.gz image", accept_multiple_files=True)


# Verifica si se cargaron archivos
if uploaded_files is not None:
    for uploaded_file in uploaded_files:

        filename = uploaded_file.name
        destination_path = os.path.join("uploaded_images", filename)
        with open(destination_path, "wb") as f:
            f.write(uploaded_file.getbuffer())     


################################# vizualizar imagenes 

folder_path = "uploaded_images"  # Ruta de la carpeta que deseas listar
st.markdown(
    """
    ### Archivos almacenados
    """
)

nombreImagenes = ["Selecciona una imagen"]
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

option = st.selectbox('Selecciona una opción', nombreImagenes, index=0)
valor_minimo = 0

if option != "Selecciona una imagen":

    path = folder_path+"/"+option
    image_data = nib.load(folder_path+"/"+option).get_fdata()  
    
    fig, ax = plt.subplots(figsize=(50, 50))

    if option == 'T1.nii.gz':
        opciones = ['Ninguna', 'Axial', 'Sagital', 'Coronal']
        corte = st.radio('Selecciona una opción', opciones)

        if corte == 'Axial':
            valor_especifico = 100
            valor_maximo = image_data.shape[1]
            ####slider
            valor_seleccionado = st.slider("Selecciona un valor", valor_minimo, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(image_data[:, valor_seleccionado, :], cmap='bone')

        if corte == 'Sagital':
            valor_especifico = 100
            valor_maximo = image_data.shape[2]
            ####slider
            valor_seleccionado = st.slider("Selecciona un valor", valor_minimo, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(np.rot90(image_data[:, :, valor_seleccionado],k=-1), cmap='bone')

        if corte == 'Coronal':
            valor_especifico = 100
            valor_maximo = image_data.shape[0]
            ####slider
            valor_seleccionado = st.slider("Selecciona un valor", valor_minimo, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(image_data[valor_seleccionado, :, :], cmap='bone')

    if option == 'FLAIR.nii.gz' or option == 'IR.nii.gz' :
            valor_especifico = 25
            valor_maximo = image_data.shape[2]
            ####slider
            valor_seleccionado = st.slider("Selecciona un valor", valor_minimo, valor_maximo, valor_especifico)
            ###imagen
            ax.imshow(np.rot90(image_data[:, :, valor_seleccionado], k=-1), cmap='bone')

    #ax.axis('off')  # Opcional: para ocultar los ejes 
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Mostrar la imagen utilizando st.image
    st.image(buffer, caption = option, use_column_width=True)
    #st.pyplot(fig)
    

################################ Boton de eliminar imagenes
st.sidebar.write("Para borrar al instante de haberlos subido se debe recargar la pagina")
if st.sidebar.button("Borrar archivos"):
    # Verifica si la carpeta existe
    if os.path.exists(folder_path):
        # Enumera los archivos en la carpeta
        files = os.listdir(folder_path)

        # Itera sobre cada archivo y lo borra
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
        
        st.success("Archivos borrados correctamente")
        st.experimental_rerun()
    else:
        st.error(f"La carpeta '{folder_path}' no existe")