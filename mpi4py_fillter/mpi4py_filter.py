# process_images_mpi.py
from mpi4py import MPI
import cv2
import numpy as np
import sys
import os


# Función para aplicar el filtro a varias imagenes
def apply_filter(image_paths, kernel, output_folder):
    # Crea la carpeta de salida
    os.makedirs(output_folder, exist_ok=True)

    for path in image_paths:
        original_image = cv2.imread(path)

        if original_image is None:
            print("Error al abrir imagen: ", path, "\n")
            continue

        filtered_image = cv2.filter2D(original_image, -1, kernel)

        # Guardar la imagen filtrada en la carpeta de salida
        filename = os.path.basename(path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, filtered_image)


# Función para procesar una lista de imágenes de forma concurrente
def mpi4py_process(image_paths, kernel, output_folder):
    apply_filter(image_paths, kernel, output_folder)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Obteniendo image_paths, kernel_str y output_folder de los argumentos de línea de comandos
    image_paths = sys.argv[1:-2]
    kernel_str = sys.argv[-2]
    output_folder = sys.argv[-1]

    # Convierte la cadena del kernel a un arreglo NumPy
    kernel = np.fromstring(kernel_str.replace("[", "").replace("]", ""), sep=",")

    mpi4py_process(image_paths, kernel, output_folder)
