
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

const int WIDTH = 1024;    // ширина изображения
const int HEIGHT = 4096;   // высота изображения

// Генерация случайного изображения
void generateRandomImage(int* image, int height, int width) {
    for (int i = 0; i < height * width; ++i) {
        image[i] = rand() % 2;
    }
}

// Простое ядро без shared memory
__global__ void findWidthsSimple(const int* image, int* widths, int width, int height) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    int left = -1, right = -1;
    for (int x = 0; x < width; ++x) {
        if (image[y * width + x] == 1) {
            if (left == -1) left = x;
            right = x;
        }
    }
    if (left == -1 || right == -1)
        widths[y] = 0;
    else
        widths[y] = right - left + 1;
}

// Оптимизированное ядро с shared memory
__global__ void findWidthsShared(const int* image, int* widths, int width, int height) {
    extern __shared__ int shared[];  // shared memory для одной строки

    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    int tx = threadIdx.x;

    // Копирование строки в shared memory
    for (int i = tx; i < width; i += blockDim.x) {
        shared[i] = image[y * width + i];
    }
    __syncthreads();  // обязательно синхронизация!

    int left = -1, right = -1;
    for (int x = 0; x < width; ++x) {
        if (shared[x] == 1) {
            if (left == -1) left = x;
            right = x;
        }
    }

    if (left == -1 || right == -1)
        widths[y] = 0;
    else
        widths[y] = right - left + 1;
}

// Проверка правильности
bool checkResults(int* a, int* b, int size) {
    for (int i = 0; i < size; ++i) {
        if (a[i] != b[i]) {
            std::cout << "Mismatch at " << i << ": " << a[i] << " != " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    // Выделяем память на CPU
    int* h_image = new int[HEIGHT * WIDTH];
    int* h_widthsSimple = new int[HEIGHT];
    int* h_widthsShared = new int[HEIGHT];

    generateRandomImage(h_image, HEIGHT, WIDTH);

    // Выделяем память на GPU
    int* d_image, * d_widthsSimple, * d_widthsShared;
    cudaMalloc(&d_image, HEIGHT * WIDTH * sizeof(int));
    cudaMalloc(&d_widthsSimple, HEIGHT * sizeof(int));
    cudaMalloc(&d_widthsShared, HEIGHT * sizeof(int));

    cudaMemcpy(d_image, h_image, HEIGHT * WIDTH * sizeof(int), cudaMemcpyHostToDevice);

    // Запуск первого ядра (простого)
    dim3 blockSize(256);
    dim3 gridSize((HEIGHT + blockSize.x - 1) / blockSize.x);

    cudaEvent_t startSimple, stopSimple;
    cudaEventCreate(&startSimple);
    cudaEventCreate(&stopSimple);

    cudaEventRecord(startSimple);
    findWidthsSimple << <gridSize, blockSize >> > (d_image, d_widthsSimple, WIDTH, HEIGHT);
    cudaEventRecord(stopSimple);

    cudaMemcpy(h_widthsSimple, d_widthsSimple, HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stopSimple);
    float msSimple = 0;
    cudaEventElapsedTime(&msSimple, startSimple, stopSimple);
    std::cout << "Simple kernel time: " << msSimple << " ms\n";

    // Запуск второго ядра (с shared memory)
    cudaEvent_t startShared, stopShared;
    cudaEventCreate(&startShared);
    cudaEventCreate(&stopShared);

    cudaEventRecord(startShared);
    findWidthsShared << <gridSize, blockSize, WIDTH * sizeof(int) >> > (d_image, d_widthsShared, WIDTH, HEIGHT);
    cudaEventRecord(stopShared);

    cudaMemcpy(h_widthsShared, d_widthsShared, HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stopShared);
    float msShared = 0;
    cudaEventElapsedTime(&msShared, startShared, stopShared);
    std::cout << "Shared memory kernel time: " << msShared << " ms\n";

    // Проверка правильности
    if (checkResults(h_widthsSimple, h_widthsShared, HEIGHT)) {
        std::cout << "Results are correct.\n";
    }
    else {
        std::cout << "Results are incorrect!\n";
    }

    // Освобождение ресурсов
    cudaFree(d_image);
    cudaFree(d_widthsSimple);
    cudaFree(d_widthsShared);
    delete[] h_image;
    delete[] h_widthsSimple;
    delete[] h_widthsShared;

    cudaEventDestroy(startSimple);
    cudaEventDestroy(stopSimple);
    cudaEventDestroy(startShared);
    cudaEventDestroy(stopShared);

    // Чтобы консоль не закрывалась
    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.get();

    return 0;
}