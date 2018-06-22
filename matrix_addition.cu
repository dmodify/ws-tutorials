// Sample codes for tutorial 3
#include <stdlib.h>
#include <stdio.h>

// the kernel computes the matrix addition C = A + B
// each thread performs one pair-wise addition
__global__ void matrixAdd(int *d_x, int *d_y, int *d_z, int n)
{
	// compute the global element index in 2D this thread should process
	int idx_x = (blockDim.x * blockIdx.x) + threadIdx.x;
	int idx_y = (blockDim.y * blockIdx.y) + threadIdx.y;

	// Addressing element of 2D array in 1D with thread index in 2D 
	int array_idx = (idx_x * n) + idx_y;

	if (array_idx < n * n) {
		*(d_z + array_idx) = (*(d_x + array_idx)) + (*(d_y + array_idx));
	}
}

int main()
{
	// define the number of elements in x-dimension & y-dimension the arrays
	size_t n = 1<<10;

	// compute the size of the 2D arrays in bytes
	size_t size_bytes = n * n * sizeof(int);

	// initialize 2D input array & allocate 2D output array on host 
	int h_x[n][n];  // matrix X
	int h_y[n][n];  // matrix Y
	int h_z[n][n];  // matrix Z
	// intialize data on host
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<n; j++) {
			h_x[i][j] = 0;
			h_y[i][j] = -1; // with values of 1 & -1, elements in Z should be all -1s
		}
	}

	// declare pointers to device arrays
	int *d_x = 0;
	int *d_y = 0;
	int *d_z = 0;

	// allocate memory on the device
	cudaMalloc((void**) &d_x, size_bytes);
	cudaMalloc((void**) &d_y, size_bytes);
	cudaMalloc((void**) &d_z, size_bytes);

	//copy input data to device memory
	cudaMemcpy(d_x, &h_x[0][0], size_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, &h_y[0][0], size_bytes, cudaMemcpyHostToDevice);


	// compute grid size
	const size_t block_size_1d = 16;
    size_t grid_size_1d = n / block_size_1d;
      // deal with a possible partial final block
    if(n % block_size_1d) 
    	++grid_size_1d;

    dim3 dimGrid(grid_size_1d, grid_size_1d, 1);
	dim3 dimBlock(block_size_1d, block_size_1d, 1);

    // define CUDA timers
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start CUDA timers
    cudaEventRecord(start, 0);

    // launch the kernel
	matrixAdd<<<dimGrid, dimBlock>>>(d_x, d_y, d_z, n);

	// stop CUDA timers & sync stop time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// compute elpased time
	cudaEventElapsedTime(&time, start, stop);

	// copy the result back to the host memory
	cudaMemcpy(&h_z[0][0], d_z, size_bytes, cudaMemcpyDeviceToHost);

	// check whether the result is correct
	int i,j;
  	for(i = 0; i < n; i++) {
  		for(j = 0; j < n; j++){
    		if (h_z[i][j] != -1) {
      			printf("Element at [%d, %d] : %d != -1\n", i, j, h_z[i][j]);
      			break;
    		}
    	}
    }
    if((i == n) && (j == n)) 
    	printf("SUCCESS!\n");

    //report GPU computing time
    printf ("Time for the kernel: %f ms\n", time);

	//Free device memory
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);

	return 0;
}