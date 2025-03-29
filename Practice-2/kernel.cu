#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void gammaCorrectionKernel(const uchar4* __restrict__ input, uchar4* __restrict__ output, int width, int height, float gamma) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = y * width + x;
		uchar4 pixel = input[idx];

		float3 normalized = make_float3(pixel.x / 255.0f, pixel.y / 255.0f, pixel.z / 255.0f);
		float3 corrected = make_float3(powf(normalized.x, gamma), powf(normalized.y, gamma), powf(normalized.z, gamma));

		output[idx] = make_uchar4(
			(unsigned char)(corrected.x * 255.0f),
			(unsigned char)(corrected.y * 255.0f),
			(unsigned char)(corrected.z * 255.0f),
			255
		);
	}
}

int main() {
	int width, height, channels;
	unsigned char* h_input = stbi_load("input.png", &width, &height, &channels, 3);
	if (!h_input) {
		fprintf(stderr, "Не удалось загрузить изображение\n");
		return 1;
	}

	int pitch = (width * 3 + 3) & ~3;
	size_t imageSize = pitch * height;

	unsigned char* h_padded = (unsigned char*)malloc(imageSize);
	for (int y = 0; y < height; y++) {
		memcpy(h_padded + y * pitch, h_input + y * width * 3, width * 3);
	}

	uchar4* d_input, * d_output;
	cudaMalloc((void**)&d_input, imageSize);
	cudaMalloc((void**)&d_output, imageSize);

	cudaMemcpy2D(d_input, pitch, h_padded, pitch, width * 3, height, cudaMemcpyHostToDevice);

	dim3 blockSize(32, 8);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	float gamma = 2.2f;
	gammaCorrectionKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, gamma);
	cudaDeviceSynchronize();

	cudaMemcpy2D(h_padded, pitch, d_output, pitch, width * 3, height, cudaMemcpyDeviceToHost);

	for (int y = 0; y < height; y++) {
		memcpy(h_input + y * width * 3, h_padded + y * pitch, width * 3);
	}
	stbi_write_png("output.png", width, height, 3, h_input, width * 3);

	cudaFree(d_input);
	cudaFree(d_output);
	stbi_image_free(h_input);
	free(h_padded);

	return 0;
}
