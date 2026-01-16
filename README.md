# cuda-programming


Steps to run in colab
1 Check for GPU runtime using !nvidia-smi
2 Copy the files in the cell
3 Add magic function " %%writefile hello.cu " on top the CUDA program
4 Run the CUDA program using "!nvcc -arch=sm_75 hello.cu -o hello && ./hello" in Colab, change hello with [filename].cu


Performance tips
- Threads in a Warp (groups of 32) are indexed by .x first, then .y, then .z.

- If you use threadIdx.x, the threads in a warp access memory addresses that are right next to each other (Coalesced Access).

- If you use threadIdx.y, depending on your indexing math, you might accidentally cause "strided" memory access, which is slower.