# cuda-programming


Steps to run in colab
1 Check for GPU runtime using !nvidia-smi
2 Copy the files in the cell
3 Add magic function " %%writefile hello.cu " on top the CUDA program
4 Run the CUDA program using "!nvcc -arch=sm_75 hello.cu -o hello && ./hello" in Colab, change hello with [filename].cu