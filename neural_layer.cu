%%writefile neural_layer.cu
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

// matrix vector multiplication kernel
__global__ void matMulKernel(float* W, float* X, float* Y, int width, int height) {
    
    // calculating only along the X direction, x axis, row is constant -> using X thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            sum += W[row * width + k] * X[k];
        }
        Y[row] = sum;
    }
}

// ReLU activation kernel -> f(x) = max(0, x)
__global__ void reluKernel(float* Y, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        Y[i] = (Y[i] > 0.0f) ? Y[i] : 0.0f;
    }
}

int main() {
    // making 1024 x 1024 matrix
    const int ROWS = 1024;
    const int COLS = 1024;
    size_t w_size = ROWS * COLS * sizeof(float);
    size_t x_size = COLS * sizeof(float);
    size_t y_size = ROWS * sizeof(float);

    
    float *h_W = (float*)malloc(w_size); // (1024, 1024)
    float *h_X = (float*)malloc(x_size); // (1024, 1)
    float *h_Y = (float*)malloc(y_size);

    // initialize Matrix 
    for (int i = 0; i < ROWS * COLS; i++) h_W[i] = 0.01f;
    for (int i = 0; i < COLS; i++) h_X[i] = -5.0f;


    float *d_W, *d_X, *d_Y;
    cudaMalloc(&d_W, w_size);
    cudaMalloc(&d_X, x_size);
    cudaMalloc(&d_Y, y_size);

    cudaMemcpy(d_W, h_W, w_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, x_size, cudaMemcpyHostToDevice);

    // using "dim3" as we are working in 2D, for 2d matrix
    // as kernel is calculating in X-axis, using X thread, we only need Y-threads
    // essentially the memory is in 1D, but dim3 makes the row, col calc more easier,
    // otherweise we have to put logic for skipping elements in Y direction to move to next row

    dim3 threadsPerBlock(1, 256); // 256 threads in Y direction
    dim3 blocksPerGrid(1, (ROWS + 255) / 256);
    
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_W, d_X, d_Y, COLS, ROWS);
    cudaDeviceSynchronize();


    int threads1D = 256;
    int blocks1D = (ROWS + threads1D - 1) / threads1D;
    
    reluKernel<<<blocks1D, threads1D>>>(d_Y, ROWS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Y, d_Y, y_size, cudaMemcpyDeviceToHost);

    printf("First 5 results: %f %f %f %f %f\n", h_Y[0], h_Y[1], h_Y[2], h_Y[3], h_Y[4]);
    printf("(Expected 0.000000 because of ReLU on negative inputs)\n");

    cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);
    free(h_W); free(h_X); free(h_Y);

    return 0;
}