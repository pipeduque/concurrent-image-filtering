import streamlit as st
import os
import cv2
import time
import numpy as np
import subprocess
import threading
import multiprocessing
from simple_image_download import simple_image_download

laplace = "laplace"
square_3x3 = "square_3x3"
square_5x5 = "square_5x5"
first_class_1 = "first_class_1"
first_class_2 = "first_class_2"
first_class_3 = "first_class_3"
first_edge_3x3 = "first_edge_3x3"
first_edge_5x5 = "first_edge_5x5"
sobel_vertical = "sobel_vertical"
sobel_horizontal = "sobel_horizontal"
prewitt_vertical = "prewitt_vertical"
prewitt_horizontal = "prewitt_horizontal"

python_option = "python"
multiprocessing_option = "multiprocessing"
mpi4py_option = "mpi4py"
c_option = "c"
openmp_option = "openmp"


def download_images(query, start_index, num_images):
    response = simple_image_download.simple_image_download
    downloader = response()

    for i in range(start_index, start_index + num_images):
        downloader.download(query, i)


def download_images_main(query, num_images_per_thread=100):
    num_threads = 10
    threads = []

    for i in range(num_threads):
        start_index = i * num_images_per_thread
        thread = threading.Thread(
            target=download_images,
            name=f"Hilo {i + 1}",
            args=(query, start_index, num_images_per_thread),
        )
        threads.append(thread)
        thread.start()

    # Esperar a que terminen todos los hilos
    for thread in threads:
        thread.join()


# Función para aplicar el filtro a varias imagenes
def python_filter(image_paths, kernel, output_folder):
    # Crea la carpeta de salida
    os.makedirs(output_folder, exist_ok=True)

    for path in image_paths:
        original_image = cv2.imread(path)

        if original_image is None:
            continue

        filtered_image = cv2.filter2D(original_image, -1, kernel)

        # Guardar la imagen filtrada en la carpeta de salida
        filename = os.path.basename(path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, filtered_image)


# Función para procesar una lista de imágenes de forma concurrente
def mpi4py_filter(image_paths, kernel, num_processes, output_directory):
    # Serializa el arreglo NumPy a una cadena
    kernel_str = np.array2string(kernel, separator=",")

    # Ejecuta el script MPI con mpiexec y pasa image_paths y kernel_str como argumentos
    result = subprocess.run(
        [
            "mpiexec",
            "-n",
            str(num_processes),
            "python",
            "./mpi4py_fillter/mpi4py_filter.py",
        ]
        + image_paths
        + [kernel_str, output_directory],
        stdout=subprocess.PIPE,
        text=True,
    )

    # Obtén los resultados de la salida estándar del proceso MPI
    output = result.stdout

    print(output)


def multiprocessing_filter(image_paths, kernel, num_processes, output_folder):
    # Divide la lista de imágenes entre los procesos
    chunk_size = len(image_paths) // num_processes
    processes = []

    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i < num_processes - 1 else len(image_paths)
        local_image_paths = image_paths[start_index:end_index]

        # Cada proceso ejecuta apply_filter en su sublista de imágenes
        p = multiprocessing.Process(
            target=python_filter, args=(local_image_paths, kernel, output_folder)
        )

        processes.append(p)
        p.start()

    # Espera a que todos los procesos hayan terminado
    for p in processes:
        p.join()


# Función para aplicar el filtro a una imagen
def c_filter(image_path, kernel, output_directory):
    # Definir el comando para ejecutar el programa C
    command = [
        "./c_filter/c_filter_program",  # Nombre del programa C
        image_path,  # Directorio de imágenes
        kernel,  # Tipo de kernel
        output_directory,  # Directorio de salida
    ]

    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print("Salida estándar:", result.stdout)
        print("Error estándar:", result.stderr)
    except Exception as e:
        print(f"Error al ejecutar el programa C: {e}")


def open_mp_filter(image_path, kernel, num_cores, output_directory):
    # Definir el comando para ejecutar el programa C
    command = [
        "./open_mp_filter/open_mp_filter_program",  # Nombre del programa C
        image_path,  # Directorio de imágenes
        kernel,  # Tipo de kernel
        output_directory,  # Directorio de salida
        str(num_cores),
    ]

    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print("Salida estándar:", result.stdout)
        print("Error estándar:", result.stderr)
    except Exception as e:
        print(f"Error al ejecutar el programa C: {e}")


# Configuración de Streamlit
st.title("Descarga Concurrente de Imágenes")

# Ruta del directorio que contiene las imágenes
query = st.text_input("Tema específico")

# Número de cores/hilos/procesos
num_images_per_thread = st.slider(
    "Número de imagenes por hilo (10 hilos)",
    min_value=1,
    max_value=1000,
    value=1,
)

# Botón para iniciar el procesamiento
if st.button("Iniciar Descarga"):
    download_start_time = time.time()  # Tiempo de inicio

    download_images_main(query, num_images_per_thread)

    download_end_time = time.time()  # Tiempo de finalización
    download_execution_time = download_end_time - download_start_time

    st.write(f"Tiempo de descarga: {download_execution_time} segundos")
    st.success("Decarga completada con éxito.")


# Mostrar la sección de procesamiento solo si la descarga fue exitosa

st.title("Filtrado Concurrente de Imágenes")

# Selección de tipo de filtro
kernel_type = st.selectbox(
    "Seleccione el tipo de filtro:",
    [
        first_class_1,
        first_class_2,
        first_class_3,
        square_3x3,
        first_edge_3x3,
        square_5x5,
        first_edge_5x5,
        sobel_vertical,
        sobel_horizontal,
        laplace,
        prewitt_vertical,
        prewitt_horizontal,
    ],
)

# Definición del kernel según el tipo de filtro seleccionado
if kernel_type == first_class_1:
    kernel = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
elif kernel_type == first_class_2:
    kernel = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -2, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
elif kernel_type == first_class_3:
    kernel = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, -3, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
elif kernel_type == square_3x3:
    kernel = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
elif kernel_type == first_edge_3x3:
    kernel = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
elif kernel_type == square_5x5:
    kernel = np.array(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ]
    )
elif kernel_type == first_edge_5x5:
    kernel = np.array(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
elif kernel_type == sobel_vertical:
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
elif kernel_type == sobel_horizontal:
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
elif kernel_type == laplace:
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
elif kernel_type == prewitt_vertical:
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
elif kernel_type == prewitt_horizontal:
    kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Selección del tipo de procesamiento concurrente
concurrent_type = st.radio(
    "Seleccione el tipo de procesamiento concurrente:", ["Hilos", "Procesos"]
)

# Número de cores/hilos/procesos
num_cores = st.slider(
    "Número de Cores/Hilos/Procesos:",
    min_value=1,
    max_value=os.cpu_count(),
    value=1,
)

images_directory = "./simple_images/" + query

# Ruta de salida para las imágenes filtradas
output_directory = "./outputs"

# Determinar el tipo de lenguaje/framework seleccionado
selected_language = st.selectbox(
    "Seleccione el lenguaje/framework:",
    [python_option, multiprocessing_option, mpi4py_option, c_option, openmp_option],
)

# Botón para iniciar el procesamiento
if st.button("Iniciar Procesamiento"):
    output_folder = output_directory + "/" + selected_language + "/" + kernel_type

    # Crear el directorio de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Lista de rutas de imágenes
    image_paths = [
        os.path.join(images_directory, filename)
        for filename in os.listdir(images_directory)
    ]

    start_time = time.time()  # Tiempo de inicio

    if selected_language == python_option:
        python_filter(image_paths, kernel, output_folder)

    elif selected_language == multiprocessing_option:
        multiprocessing_filter(image_paths, kernel, num_cores, output_folder)

    elif selected_language == mpi4py_option:
        mpi4py_filter(image_paths, kernel, num_cores, output_folder)

    elif selected_language == c_option:
        c_filter(images_directory, kernel_type, output_folder)

    elif selected_language == openmp_option:
        open_mp_filter(images_directory, kernel_type, num_cores, output_folder)

    end_time = time.time()  # Tiempo de finalización

    execution_time = end_time - start_time
    st.write(f"Tiempo de ejecución: {execution_time} segundos")

    # Mostrar algunas imágenes originales y filtradas en Streamlit
    for filename in os.listdir(output_folder):
        if filename.endswith(".jpg"):
            original_image_path = os.path.join(images_directory, filename)
            filtered_image_path = os.path.join(output_folder, filename)

            original_image = cv2.imread(original_image_path)
            filtered_image = cv2.imread(filtered_image_path)

            if original_image is None or filtered_image is None:
                continue

            st.image(
                cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                caption="Imagen Original",
                use_column_width=True,
            )
            st.image(
                cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB),
                caption="Imagen Filtrada",
                use_column_width=True,
            )

            # Calcular estadísticas de píxeles
            min_pixel_value = np.min(filtered_image)
            max_pixel_value = np.max(filtered_image)
            mean_pixel_value = np.mean(filtered_image)
            std_pixel_value = np.std(filtered_image)

            st.write(f"Dimensiones de la imagen filtrada: {filtered_image.shape}")
            st.write(f"Valor mínimo de píxel: {min_pixel_value}")
            st.write(f"Valor máximo de píxel: {max_pixel_value}")
            st.write(f"Valor medio de píxeles: {mean_pixel_value}")
            st.write(f"Desviación estándar de píxeles: {std_pixel_value}")

    st.success("Procesamiento completado con éxito.")
