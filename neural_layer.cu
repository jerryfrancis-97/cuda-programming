%%writefile neural_layer.cu
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

// matrix vector multiplication kernel
__global__ void matMulKernel(float* W, float* X, float* Y, float* B, int width, int height) {
    
    // calculating only along the X direction, x axis, row is constant -> using Y thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            sum += W[row * width + k] * X[k];
        }

        float val = sum + B[row]; // adding bias      
        val = val > 0 ? val : 0; // applying RELU
        Y[row] = val;

    }


}



int main() {
    // making 1024 x 1024 matrix
    const int ROWS = 1024;
    const int COLS = 1024;
    size_t w_size = ROWS * COLS * sizeof(float);
    size_t x_size = COLS * sizeof(float);
    size_t y_size = ROWS * sizeof(float);

    
    float *h_W = (float*)malloc(w_size);
    float *h_X = (float*)malloc(x_size);
    float *h_Y = (float*)malloc(y_size);
    float *h_B = (float*)malloc(y_size);

    // initialize Matrix 
    for (int i = 0; i < ROWS * COLS; i++) h_W[i] = 0.7f;
    for (int i = 0; i < COLS; i++) h_X[i] = 0.0f;

    // initialize bias
    for(int i=0; i<ROWS; i++) h_B[i] = 0.7f;

    float *d_W, *d_X, *d_Y, *d_B;
    cudaMalloc(&d_W, w_size);
    cudaMalloc(&d_X, x_size);
    cudaMalloc(&d_Y, y_size);
    cudaMalloc(&d_B, y_size);

    cudaMemcpy(d_W, h_W, w_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, y_size, cudaMemcpyHostToDevice);

    // using "dim3" as we are working in 2D, for 2d matrix
    // as kernel is calculating in X-axis, using X thread, we only need Y-threads
    // essentially the memory is in 1D, but dim3 makes the row, col calc more easier,
    // otherweise we have to put logic for skipping elements in Y direction to move to next row

    dim3 threadsPerBlock(1, 256); // 256 threads in Y direction
    dim3 blocksPerGrid(1, (ROWS + 255) / 256);
    
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_W, d_X, d_Y, d_B, COLS, ROWS);

    // check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();


    cudaMemcpy(h_Y, d_Y, y_size, cudaMemcpyDeviceToHost);

    printf("First 5 results: %f %f %f %f %f\n", h_Y[0], h_Y[1], h_Y[2], h_Y[3], h_Y[4]);
    printf("(Expected 0.7 because of bias)\n");

    cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);
    free(h_W); free(h_X); free(h_Y);

    return 0;
}