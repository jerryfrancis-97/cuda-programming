%%writefile vector_add.cu
#include <iostream>
#include <cuda_runtime.h>

// this is kernel, this runs on GPU
// doesn't have return type as all the results
// go to cudaMalloc address that CPU gave to GPU
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // get the global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // should be within N bound of array
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000; 
    size_t size = N * sizeof(float); 
    // unsigned interger type for 64-bit systems size_t =64

    // make memory for host (CPU)
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // make device memory (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // launching kernel
    // we will do ceil(N/ threads) to get blocks per grid
    // this math works as C++ truncates the decimal part for interger type
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result at index 0: " << h_C[0] << " (Expected: 3)" << std::endl;
    std::cout << "Result at index 999999: " << h_C[999999] << " (Expected: 3)" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}