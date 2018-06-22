#include <stdio.h>

//Devices with compute capability 2.x or 
//higher support calls to printf()  from within a CUDA kernel
__global__ void mykernel(void) {
	printf("Hello Parallel World! from block %d, thread %d\n", 
	blockIdx.x, threadIdx.x);
}

int main(void) {
	mykernel<<<2,2>>>();
	cudaDeviceSynchronize();
	return 0;
}