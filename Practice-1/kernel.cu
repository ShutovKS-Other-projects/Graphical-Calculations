#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Подключаем stb_image для загрузки изображения
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
// Подключаем stb_image_write для сохранения изображения
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA-ядро для гамма-коррекции
__global__ void gammaCorrectionKernel(unsigned char* input, unsigned char* output, int width, int height, float gamma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // 3 канала RGB
        for (int i = 0; i < 3; i++) {
            float normalized = input[idx + i] / 255.0f;
            float corrected = powf(normalized, gamma);
            output[idx + i] = (unsigned char)(corrected * 255.0f);
        }
    }
}

int main() {
    int width, height, channels;
    // Загрузка изображения (форсируем 3 канала, даже если оригинал другой)
    unsigned char* h_input = stbi_load("input.png", &width, &height, &channels, 3);
    if (!h_input) {
        fprintf(stderr, "Не удалось загрузить изображение\n");
        return 1;
    }

    size_t imageSize = width * height * 3 * sizeof(unsigned char);
    unsigned char* h_output = (unsigned char*)malloc(imageSize);

    // Выделяем память на устройстве
    unsigned char* d_input, * d_output;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);

    // Копируем данные изображения на GPU
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    // Настраиваем параметры ядра
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    float gamma = 2.2f; // Значение гаммы
    gammaCorrectionKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, gamma);
    cudaDeviceSynchronize();

    // Копируем результат обратно
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Сохраняем обработанное изображение (например, в формате PNG)
    stbi_write_png("output.png", width, height, 3, h_output, width * 3);

    // Освобождаем ресурсы
    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_input);
    free(h_output);

    return 0;
}
