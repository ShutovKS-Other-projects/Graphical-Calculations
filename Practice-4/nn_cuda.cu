#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>

// Размерности
const int IN_DIM = 128;
const int HIDDEN_DIM = 32;
const int OUT_DIM = 1;
const int BATCH = 8;

// CUDA check
#define CUDA_CHECK(err) if(err!=cudaSuccess){std::cerr<<cudaGetErrorString(err);return -1;}
#define CUBLAS_CHECK(err) if(err!=CUBLAS_STATUS_SUCCESS){std::cerr<<"cuBLAS error";return -1;}

__global__ void add_bias_relu(float* Z, const float* b, int batch, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch*dim) {
        int j = idx % dim;
        float v = Z[idx] + b[j];
        Z[idx] = (v > 0 ? v : 0);
    }
}

int main() {
    // RNG на CPU
    std::mt19937 gen(0);
    std::normal_distribution<float> dist(0,1);

    // Выделяем и инициализируем CPU данные
    std::vector<float> h_X(BATCH*IN_DIM), h_Y(BATCH*OUT_DIM);
    std::vector<float> h_W1(IN_DIM*HIDDEN_DIM), h_b1(HIDDEN_DIM);
    std::vector<float> h_W2(HIDDEN_DIM*OUT_DIM), h_b2(OUT_DIM);
    for (auto& x : h_X) x = dist(gen);
    for (auto& y : h_Y) y = dist(gen);
    for (auto& w : h_W1) w = dist(gen);
    for (auto& w : h_W2) w = dist(gen);

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // Выделение GPU буферов
    float *d_X, *d_Z1, *d_H, *d_Ypred;
    float *d_W1, *d_b1, *d_W2, *d_b2;
    CUDA_CHECK(cudaMalloc(&d_X, BATCH*IN_DIM*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z1, BATCH*HIDDEN_DIM*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ypred, BATCH*OUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1, IN_DIM*HIDDEN_DIM*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, HIDDEN_DIM*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, HIDDEN_DIM*OUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, OUT_DIM*sizeof(float)));
    
    // Копируем на устройство
    CUBLAS_CHECK(cublasSetVector(BATCH*IN_DIM, sizeof(float), h_X.data(), 1, d_X, 1));
    CUBLAS_CHECK(cublasSetVector(BATCH*OUT_DIM, sizeof(float), h_Y.data(), 1, d_Ypred, 1)); // reuse for Y
    CUBLAS_CHECK(cublasSetVector(IN_DIM*HIDDEN_DIM, sizeof(float), h_W1.data(), 1, d_W1, 1));
    CUBLAS_CHECK(cublasSetVector(HIDDEN_DIM, sizeof(float), h_b1.data(), 1, d_b1, 1));
    CUBLAS_CHECK(cublasSetVector(HIDDEN_DIM*OUT_DIM, sizeof(float), h_W2.data(), 1, d_W2, 1));
    CUBLAS_CHECK(cublasSetVector(OUT_DIM, sizeof(float), h_b2.data(), 1, d_b2, 1));

    // Настройка таймера
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float alpha = 1.0f, beta = 0.0f;
    std::vector<float> times;

    // 2 разогрева
    for(int i=0;i<2;i++){
        // Forward
        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            BATCH, HIDDEN_DIM, IN_DIM,
            &alpha,
            d_X, CUDA_R_32F, BATCH,
            d_W1, CUDA_R_32F, IN_DIM,
            &beta,
            d_Z1, CUDA_R_32F, BATCH,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        int threads=256; int blocks=(BATCH*HIDDEN_DIM+threads-1)/threads;
        add_bias_relu<<<blocks,threads>>>(d_Z1, d_b1, BATCH, HIDDEN_DIM);
        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            BATCH, OUT_DIM, HIDDEN_DIM,
            &alpha,
            d_Z1, CUDA_R_32F, BATCH,
            d_W2, CUDA_R_32F, HIDDEN_DIM,
            &beta,
            d_Ypred, CUDA_R_32F, BATCH,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // 10 измерений
    for(int it=0; it<10; ++it) {
        CUDA_CHECK(cudaEventRecord(start));

        // Forward
        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            BATCH, HIDDEN_DIM, IN_DIM,
            &alpha,
            d_X, CUDA_R_32F, BATCH,
            d_W1, CUDA_R_32F, IN_DIM,
            &beta,
            d_Z1, CUDA_R_32F, BATCH,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        int threads=256; int blocks=(BATCH*HIDDEN_DIM+threads-1)/threads;
        add_bias_relu<<<blocks,threads>>>(d_Z1, d_b1, BATCH, HIDDEN_DIM);
        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            BATCH, OUT_DIM, HIDDEN_DIM,
            &alpha,
            d_Z1, CUDA_R_32F, BATCH,
            d_W2, CUDA_R_32F, HIDDEN_DIM,
            &beta,
            d_Ypred, CUDA_R_32F, BATCH,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    std::cout << "CUDA + cuBLAS ms per iter:\n";
    for(auto t: times) std::cout<<t<<" ";
    std::cout<<"\n";

    // Очистка
    cudaFree(d_X); cudaFree(d_Z1); cudaFree(d_Ypred);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cublasDestroy(handle);
    return 0;
}