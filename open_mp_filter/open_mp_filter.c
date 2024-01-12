#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "../stb_image/stb_image.h"
#include "../stb_image/stb_image_write.h"
#include <omp.h>

const char* laplace = "laplace";
const char* square_3x3 = "square_3x3";
const char* square_5x5 = "square_5x5";
const char* first_class_1 = "first_class_1";
const char* first_class_2 = "first_class_2";
const char* first_class_3 = "first_class_3";
const char* first_edge_3x3 = "first_edge_3x3";
const char* first_edge_5x5 = "first_edge_5x5";
const char* sobel_vertical = "sobel_vertical";
const char* sobel_horizontal = "sobel_horizontal";
const char* prewitt_vertical = "prewitt_vertical";
const char* prewitt_horizontal = "prewitt_horizontal";

void apply_filter(const char* input_path, const char* output_path, int kernel[5][5]) {
    // Cargar la imagen
    int width, height, channels;
    unsigned char* original_image = stbi_load(input_path, &width, &height, &channels, 0);

    if (!original_image) {
        printf("Error al cargar la imagen: %s\n", input_path);
        return;
    }

    // Crear la imagen filtrada
    unsigned char* filtered_image = (unsigned char*)malloc(width * height * channels);

    // Aplicar el filtro a la imagen
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int sum = 0;
                for (int i = 0; i < 5; ++i) {
                    for (int j = 0; j < 5; ++j) {
                        int pixel_y = y + i - 2;
                        int pixel_x = x + j - 2;

                        // Manejar los bordes de la imagen
                        if (pixel_y >= 0 && pixel_y < height && pixel_x >= 0 && pixel_x < width) {
                            sum += kernel[i][j] * original_image[(pixel_y * width + pixel_x) * channels + c];
                        }
                    }
                }
                filtered_image[(y * width + x) * channels + c] = (unsigned char)sum;
            }
        }
    }

    // Crear el directorio de salida si no existe
    mkdir(output_path, 0777);

    // Obtener solo el nombre del archivo de entrada
    const char* input_filename = strrchr(input_path, '/');
    if (input_filename == NULL) {
        input_filename = input_path;
    } else {
        input_filename++;  // Salta el carácter '/'
    }

    // Construir el nombre del archivo de salida
    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "%s/%s", output_path, input_filename);

    // Guardar la imagen filtrada
    stbi_write_png(output_filename, width, height, channels, filtered_image, width * channels);

    // Liberar la memoria
    stbi_image_free(original_image);
    free(filtered_image);
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("Uso: %s <directorio_imagenes> <kernel_type> <output_directory> <num_threads>\n", argv[0]);
        return 1;
    }

    const char* images_directory = argv[1];
    const char* kernel_type = argv[2];
    const char* output_directory = argv[3];   
    const int num_threads = atoi(argv[4]);
 
    printf("%d", num_threads);

    // Definición del kernel según el tipo de filtro seleccionado
    int kernel[5][5];

    // Seleccionar el kernel según el tipo
    if (strcmp(kernel_type, first_class_1) == 0) {
        int kernel[5][5] = {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, -1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    } else if (strcmp(kernel_type, first_class_2) == 0) {
        int kernel[5][5] = {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, -2, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}};
    } else if (strcmp(kernel_type, first_class_3) == 0) {
        int kernel[5][5] = {{0, 0, -1, 0, 0}, {0, 0, 3, 0, 0}, {0, 0, -3, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}};
    } else if (strcmp(kernel_type, square_3x3) == 0) {
        int kernel[5][5] = {{0, 0, 0, 0, 0}, {0, -1, 2, -1, 0}, {0, 2, -4, 2, 0}, {0, -1, 2, -1, 0}, {0, 0, 0, 0, 0}};
    } else if (strcmp(kernel_type, first_edge_3x3) == 0) {
        int kernel[5][5] = {{0, 0, 0, 0, 0}, {0, -1, 2, -1, 0}, {0, 2, -4, 2, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    } else if (strcmp(kernel_type, square_5x5) == 0) {
        int kernel[5][5] = {{-1, 2, -2, 2, -1}, {2, -6, 8, -6, 2}, {-2, 8, -12, 8, -2}, {2, -6, 8, -6, 2}, {-1, 2, -2, 2, -1}};
    } else if (strcmp(kernel_type, first_edge_5x5) == 0) {
        int kernel[5][5] = {{-1, 2, -2, 2, -1}, {2, -6, 8, -6, 2}, {-2, 8, -12, 8, -2}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    } else if (strcmp(kernel_type, sobel_vertical) == 0) {
        int kernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    } else if (strcmp(kernel_type, sobel_horizontal) == 0) {
        int kernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    } else if (strcmp(kernel_type, laplace) == 0) {
        int kernel[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
    } else if (strcmp(kernel_type, prewitt_vertical) == 0) {
        int kernel[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    } else if (strcmp(kernel_type, prewitt_horizontal) == 0) {
        int kernel[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    }

    // Procesar cada imagen en el directorio
    DIR* dir = opendir(images_directory);
    if (dir == NULL) {
        perror("Error al abrir el directorio");
        return 1;
    }

    struct dirent* entry;

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; ++i) {
        // Cada hilo procesa una parte de las imágenes
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_REG) {  // Archivo regular
                char input_path[256];
                snprintf(input_path, sizeof(input_path), "%s/%s", images_directory, entry->d_name);

                // Aplicar el filtro y guardar la imagen filtrada
                apply_filter(input_path, output_directory, kernel);
            }
        }
    }

    closedir(dir);

    return 0;
}
