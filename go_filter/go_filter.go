package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"sync"
)

const (
	laplace           = "laplace"
	square3x3         = "square_3x3"
	square5x5         = "square_5x5"
	firstClass1       = "first_class_1"
	firstClass2       = "first_class_2"
	firstClass3       = "first_class_3"
	firstEdge3x3      = "first_edge_3x3"
	firstEdge5x5      = "first_edge_5x5"
	sobelVertical     = "sobel_vertical"
	sobelHorizontal   = "sobel_horizontal"
	prewittVertical   = "prewitt_vertical"
	prewittHorizontal = "prewitt_horizontal"
)

// applyKernel aplica un kernel a una imagen y devuelve la imagen resultante.
func applyKernel(imagen image.Image, kernel [][]float64) image.Image {
	rect := imagen.Bounds()
	nuevaImagen := image.NewRGBA(rect)

	ancho, alto := rect.Dx(), rect.Dy()

	for y := 0; y < alto; y++ {
		for x := 0; x < ancho; x++ {
			var r, g, b, a float64

			// Aplicar kernel
			for ky := 0; ky < len(kernel); ky++ {
				for kx := 0; kx < len(kernel[ky]); kx++ {
					pixelX := x + kx - len(kernel[ky])/2
					pixelY := y + ky - len(kernel)/2

					// Verificar que el pixel esté dentro de los límites de la imagen
					if pixelX >= 0 && pixelX < ancho && pixelY >= 0 && pixelY < alto {
						pixel := imagen.At(pixelX, pixelY)
						rPixel, gPixel, bPixel, aPixel := pixel.RGBA()

						// Sumar el valor del kernel multiplicado por el valor del pixel
						r += float64(kernel[ky][kx]) * float64(rPixel)
						g += float64(kernel[ky][kx]) * float64(gPixel)
						b += float64(kernel[ky][kx]) * float64(bPixel)
						a += float64(kernel[ky][kx]) * float64(aPixel)
					}
				}
			}

			// Normalizar los valores
			r = limitar(r, 0, 255)
			g = limitar(g, 0, 255)
			b = limitar(b, 0, 255)
			a = limitar(a, 0, 255)

			// Asignar el nuevo color al píxel
			nuevaImagen.SetRGBA(x, y, color.RGBA{
				R: uint8(r),
				G: uint8(g),
				B: uint8(b),
				A: uint8(a),
			})
		}
	}

	return nuevaImagen
}

// limitar limita el valor entre un rango específico.
func limitar(valor, min, max float64) float64 {
	if valor < min {
		return min
	} else if valor > max {
		return max
	}
	return valor
}

// openImage abre una imagen desde un archivo.
func openImage(ruta string) (image.Image, error) {
	archivo, err := os.Open(ruta)
	if err != nil {
		return nil, err
	}
	defer archivo.Close()

	img, _, err := image.Decode(archivo)
	if err != nil {
		return nil, err
	}

	return img, nil
}

// saveImage guarda una imagen en un archivo.
func saveImage(ruta string, img image.Image) error {
	archivo, err := os.Create(ruta)
	if err != nil {
		return err
	}
	defer archivo.Close()

	return jpeg.Encode(archivo, img, nil)
}

func applyFilter(imagePaths []string, outputPath string, kernel [][]float64, wg *sync.WaitGroup, goroutineNumber int) {
	defer wg.Done()

	fmt.Printf("Inicia Goroutine %d\n", goroutineNumber)

	// Procesar cada imagen en la lista
	for _, inputPath := range imagePaths {

		originalImage, err := openImage(inputPath)
		if err != nil {
			log.Printf("Error al abrir la imagen %s\n", inputPath)
			continue

		}

		filterImage := applyKernel(originalImage, kernel)

		// Construir la ruta del archivo de salida
		outputFileName := fmt.Sprintf("%s/%s", outputPath, filepath.Base(inputPath))

		saveImage(outputFileName, filterImage)
	}

	// Imprimir el número de gorutina
	fmt.Printf("Finaliza Goroutine %d\n", goroutineNumber)
}

func main() {
	if len(os.Args) != 5 {
		fmt.Printf("Uso: %s <directorio_imagenes> <kernel_type> <output_directory>\n", os.Args[0])
		os.Exit(1)
	}

	imagesDirectory := os.Args[1]
	kernelType := os.Args[2]
	outputDirectory := os.Args[3]
	numRoutines, _ := strconv.Atoi(os.Args[4])

	// Definición del kernel según el tipo de filtro seleccionado
	var kernel [][]float64

	// Seleccionar el kernel según el tipo
	switch kernelType {
	case firstClass1:
		kernel = [][]float64{{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, -1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}
	case firstClass2:
		kernel = [][]float64{{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, -2, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}}
	case firstClass3:
		kernel = [][]float64{{0, 0, -1, 0, 0}, {0, 0, 3, 0, 0}, {0, 0, -3, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}}
	case square3x3:
		kernel = [][]float64{{0, 0, 0, 0, 0}, {0, -1, 2, -1, 0}, {0, 2, -4, 2, 0}, {0, -1, 2, -1, 0}, {0, 0, 0, 0, 0}}
	case firstEdge3x3:
		kernel = [][]float64{{0, 0, 0, 0, 0}, {0, -1, 2, -1, 0}, {0, 2, -4, 2, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}
	case square5x5:
		kernel = [][]float64{{-1, 2, -2, 2, -1}, {2, -6, 8, -6, 2}, {-2, 8, -12, 8, -2}, {2, -6, 8, -6, 2}, {-1, 2, -2, 2, -1}}
	case firstEdge5x5:
		kernel = [][]float64{{-1, 2, -2, 2, -1}, {2, -6, 8, -6, 2}, {-2, 8, -12, 8, -2}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}
	case sobelVertical:
		kernel = [][]float64{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}
	case sobelHorizontal:
		kernel = [][]float64{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}
	case laplace:
		kernel = [][]float64{{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}}
	case prewittVertical:
		kernel = [][]float64{{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}}
	case prewittHorizontal:
		kernel = [][]float64{{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}}
	default:
		log.Println("Tipo de kernel no reconocido.")
		os.Exit(1)
	}

	// Crear el directorio de salida si no existe
	err := os.MkdirAll(outputDirectory, os.ModePerm)
	if err != nil {
		log.Fatalf("Error al crear el directorio de salida: %v\n", err)
	}

	// Obtener la lista de archivos en el directorio de imágenes
	files, err := os.ReadDir(imagesDirectory)
	if err != nil {
		log.Fatalf("Error al leer el directorio de imágenes: %v\n", err)
	}

	// Dividir la lista de imágenes entre los procesos
	chunkSize := (len(files) + numRoutines - 1) / numRoutines
	var wg sync.WaitGroup

	// Variable para identificar cada gorutina
	var goroutineNumber int

	// Procesar cada "chunk" en paralelo utilizando gorutinas
	for i := 0; i < numRoutines; i++ {
		startIndex := i * chunkSize
		endIndex := (i + 1) * chunkSize
		if endIndex > len(files) {
			endIndex = len(files)
		}

		localImagePaths := make([]string, endIndex-startIndex)
		for j, file := range files[startIndex:endIndex] {
			localImagePaths[j] = filepath.Join(imagesDirectory, file.Name())
		}

		// Incrementar el contador del grupo de espera
		wg.Add(1)

		// Incrementar el número de gorutina
		goroutineNumber++

		// Utilizar una gorutina para procesar el "chunk"
		go applyFilter(localImagePaths, outputDirectory, kernel, &wg, goroutineNumber)
	}

	wg.Wait()
}
