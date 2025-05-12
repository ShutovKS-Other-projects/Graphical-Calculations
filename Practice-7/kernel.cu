#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

// CUDA-ядро: считает Gx, Gy и их модуль
__global__ void sobelKernel(
	const unsigned char* __restrict__ gray,
	unsigned char* __restrict__ edges,
	int width, int height
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

	int idx = y * width + x;

	// Собель по X
	int gx =
		-gray[(y - 1) * width + (x - 1)] + gray[(y - 1) * width + (x + 1)]
		- 2 * gray[y * width + (x - 1)] + 2 * gray[y * width + (x + 1)]
		- gray[(y + 1) * width + (x - 1)] + gray[(y + 1) * width + (x + 1)];

	// Собель по Y
	int gy =
		-gray[(y - 1) * width + (x - 1)] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + (x + 1)]
		+ gray[(y + 1) * width + (x - 1)] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + (x + 1)];

	int mag = sqrtf(float(gx * gx + gy * gy));
	if (mag > 255) mag = 255;
	edges[idx] = static_cast<unsigned char>(mag);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>\n";
		return 1;
	}
	const char* inPath = argv[1];
	const char* outPath = argv[2];

	// 1) Читаем картинку
	cv::Mat img = cv::imread(inPath, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cerr << "Can't load image: " << inPath << "\n";
		return 1;
	}
	int width = img.cols;
	int height = img.rows;
	size_t numPixels = width * height;
	size_t bytes = numPixels * sizeof(unsigned char);

	// 2) Выделяем память на GPU
	unsigned char* d_gray = nullptr, * d_edges = nullptr;
	cudaMalloc(&d_gray, bytes);
	cudaMalloc(&d_edges, bytes);

	// 3) Копируем серый на устройство
	cudaMemcpy(d_gray, img.data, bytes, cudaMemcpyHostToDevice);

	// 4) Запускаем ядро
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x,
		(height + block.y - 1) / block.y);
	sobelKernel << <grid, block >> > (d_gray, d_edges, width, height);
	cudaDeviceSynchronize();

	// 5) Читаем результат обратно
	std::vector<unsigned char> h_edges(numPixels);
	cudaMemcpy(h_edges.data(), d_edges, bytes, cudaMemcpyDeviceToHost);

	// 6) Сохраняем
	cv::Mat outImg(height, width, CV_8UC1, h_edges.data());
	cv::imwrite(outPath, outImg);

	// 7) Освобождаем
	cudaFree(d_gray);
	cudaFree(d_edges);

	return 0;
}
