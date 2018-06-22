// Sample codes for tutorial 2
#include <stdlib.h>
#include <stdio.h>

// the kernel computes the vector addition C = A + B
// each thread performs one pair-wise addition
__global__ void vectorAdd(int *d_a, int *d_b, int *d_c, int n)
{
  // compute the global element index this thread should process
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

  // avoid accessing out of bounds elements
  if(idx < n) {
    // sum the elements
    d_c[idx] = d_a[idx] + d_b[idx];
  }
}

int main(void)
{
  // define the number of elements in the arrays
  size_t n = 1<<20;

  // compute the size of the arrays in bytes
  size_t n_bytes = n * sizeof(int);

  // declare pointers to host & device arrays
  int *d_a = 0;
  int *d_b = 0;
  int *d_c = 0;
  int *h_a = 0;
  int *h_b = 0;
  int *h_c = 0;

  // allocate the host arrays
  h_a = (int*) malloc(n_bytes);
  h_b = (int*) malloc(n_bytes);
  h_c = (int*) malloc(n_bytes);

  // allocate the device arrays
  cudaMalloc((void**) &d_a, n_bytes);
  cudaMalloc((void**) &d_b, n_bytes);
  cudaMalloc((void**) &d_c, n_bytes);

  // if any memory allocation failed, report an error message
  if(h_a == 0 || h_b == 0 || h_c == 0 ||
     d_a == 0 || d_b == 0 || d_c == 0)
  {
    printf("couldn't allocate memory\n");
    return 1;
  }

  // initialize host_array_a & host_array_b
  for(int i = 0; i < n; ++i)
  {
    // make array A a incremental linear ramp 
    h_a[i] = i;
    // make array B a revere of A
    h_b[i] = (n-1)-i;
  }

  // copy arrays a & b to the device memory
  cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice);

  // compute grid size
  const size_t block_size = 512;
  size_t grid_size = n / block_size;
  // deal with a possible partial final block
  if(n % block_size) 
    ++grid_size;

  // define CUDA timers
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start CUDA timers
  cudaEventRecord(start, 0);

  // launch the kernel
  vectorAdd<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

  // stop CUDA timers & sync stop time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // compute elpased time
  cudaEventElapsedTime(&time, start, stop);

  // copy the result back to the host memory
  cudaMemcpy(h_c, d_c, n_bytes, cudaMemcpyDeviceToHost);

  // check whether the result is correct
  int i;
  for(i= 0; i < n; i++) {
      if (h_c[i] != h_a[i] + h_b[i]) {
        printf("In correct result at %d: %d + %d = %d\n", i, h_a[i], h_b[i], h_c[i]);
        break;
      }
  }
  if((i == n)) 
    printf("SUCCESS!\n");

  //report GPU computing time
  printf ("Time for the kernel: %f ms\n", time);

  // deallocate host memory
  free(h_a);
  free(h_b);
  free(h_c);

  // deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}