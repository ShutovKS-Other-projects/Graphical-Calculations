#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

// Простой подход (без разделяемой памяти)
__global__ void matrixSquareSimple(float* input, float* output, int N) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int matrixSize = N * N;
	float* in = input + tid * matrixSize;
	float* out = output + tid * matrixSize;

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			float sum = 0.0f;
			for (int k = 0; k < N; ++k) {
				sum += in[i * N + k] * in[k * N + j];
			}
			out[i * N + j] = sum;
		}
	}
}

// Оптимизированный подход (с разделяемой памятью)
__global__ void matrixSquareShared(float* input, float* output, int N) {
	extern __shared__ float s_data[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int matrixSize = N * N;

	float* in = input + tid * matrixSize;
	float* out = output + tid * matrixSize;

	float* s_matrix = s_data + threadIdx.x * matrixSize;
	for (int i = 0; i < matrixSize; ++i) {
		s_matrix[i] = in[i];
	}

	__syncthreads();

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			float sum = 0.0f;
			for (int k = 0; k < N; ++k) {
				sum += s_matrix[i * N + k] * s_matrix[k * N + j];
			}
			out[i * N + j] = sum;
		}
	}
}

//Генерация данных и проверка
void generateRandomMatrices(float* matrices, int numMatrices, int N) {
	for (int m = 0; m < numMatrices; ++m) {
		for (int i = 0; i < N * N; ++i) {
			matrices[m * N * N + i] = static_cast<float>(rand()) / RAND_MAX;
		}
	}
}

void computeMatrixSquareCPU(float* input, float* output, int N, int numMatrices) {
	for (int m = 0; m < numMatrices; ++m) {
		float* in = input + m * N * N;
		float* out = output + m * N * N;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				float sum = 0.0f;
				for (int k = 0; k < N; ++k) {
					sum += in[i * N + k] * in[k * N + j];
				}
				out[i * N + j] = sum;
			}
		}
	}
}

bool verifyResults(float* gpuResult, float* cpuResult, int numElements, float epsilon = 1e-3) {
	for (int i = 0; i < numElements; ++i) {
		if (fabs(gpuResult[i] - cpuResult[i]) > epsilon) {
			printf("Mismatch at index %d: GPU %f vs CPU %f\n", i, gpuResult[i], cpuResult[i]);
			return false;
		}
	}
	return true;
}

int main() {
	int N = 5; // Размер матрицы
	int numMatrices = 1000;
	size_t matrixSizeBytes = N * N * sizeof(float);
	size_t totalSizeBytes = numMatrices * N * N * sizeof(float);

	// Выделение памяти на хосте
	float* h_input = (float*)malloc(totalSizeBytes);
	float* h_output_simple = (float*)malloc(totalSizeBytes);
	float* h_output_shared = (float*)malloc(totalSizeBytes);
	float* h_cpu = (float*)malloc(totalSizeBytes);

	generateRandomMatrices(h_input, numMatrices, N);
	computeMatrixSquareCPU(h_input, h_cpu, N, numMatrices);

	// Выделение памяти на устройстве
	float* d_input, * d_output_simple, * d_output_shared;
	cudaMalloc(&d_input, totalSizeBytes);
	cudaMalloc(&d_output_simple, totalSizeBytes);
	cudaMalloc(&d_output_shared, totalSizeBytes);
	cudaMemcpy(d_input, h_input, totalSizeBytes, cudaMemcpyHostToDevice);

	// Настройка событий CUDA для замера времени
	cudaEvent_t startSimple, stopSimple, startShared, stopShared;
	cudaEventCreate(&startSimple);
	cudaEventCreate(&stopSimple);
	cudaEventCreate(&startShared);
	cudaEventCreate(&stopShared);
	float timeSimple = 0, timeShared = 0;

	// Запуск простого ядра
	int threadsPerBlock = 256;
	int blocks = (numMatrices + threadsPerBlock - 1) / threadsPerBlock;

	cudaEventRecord(startSimple);
	matrixSquareSimple << <blocks, threadsPerBlock >> > (d_input, d_output_simple, N);
	cudaEventRecord(stopSimple);
	cudaEventSynchronize(stopSimple);
	cudaEventElapsedTime(&timeSimple, startSimple, stopSimple);

	cudaMemcpy(h_output_simple, d_output_simple, totalSizeBytes, cudaMemcpyDeviceToHost);

	// Запуск оптимизированного ядра
	size_t sharedMemSize = threadsPerBlock * N * N * sizeof(float); // Разделяемая память на блок

	cudaEventRecord(startShared);
	matrixSquareShared << <blocks, threadsPerBlock, sharedMemSize >> > (d_input, d_output_shared, N);
	cudaEventRecord(stopShared);
	cudaEventSynchronize(stopShared);
	cudaEventElapsedTime(&timeShared, startShared, stopShared);

	cudaMemcpy(h_output_shared, d_output_shared, totalSizeBytes, cudaMemcpyDeviceToHost);

	// Проверка результатов
	bool correctSimple = verifyResults(h_output_simple, h_cpu, numMatrices * N * N);
	bool correctShared = verifyResults(h_output_shared, h_cpu, numMatrices * N * N);

	// Вывод результатов
	printf("Simple kernel: %s | Time: %.3f ms\n", correctSimple ? "Correct" : "Incorrect", timeSimple);
	printf("Shared kernel: %s | Time: %.3f ms\n", correctShared ? "Correct" : "Incorrect", timeShared);
	printf("Shared memory per block: %.2f KB\n", sharedMemSize / 1024.0);

	// Освобождение ресурсов
	cudaEventDestroy(startSimple);
	cudaEventDestroy(stopSimple);
	cudaEventDestroy(startShared);
	cudaEventDestroy(stopShared);
	// ... [Очистка памяти] ...

	return 0;
}