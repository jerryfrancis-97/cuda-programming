%%writefile hello.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello from GPU thread %d!\n \n", threadIdx.x);
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("Error: No CUDA-capable GPU found!\n");
        return 1;
    }

    printf("Starting Kernel...\n");
    # calling 1 block of 5 threads
    helloFromGPU<<<1, 5>>>();
    
    // Crucial for Colab output
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    printf("Kernel Finished.\n");

    cudaDeviceSynchronize();
    return 0;
}