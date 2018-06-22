#include <stdlib.h>
#include <stdio.h>

// define the radius, block size
#define RADIUS 3
#define BLOCK_SIZE 512

// the kernel computes the 1D stencil pattern
__global__ void fast_stencil(int *d_in, int *d_out, int n)
{
  __shared__ int temp[BLOCK_SIZE + (2 * RADIUS)];
  int g_idx = (blockDim.x * blockIdx.x)+ threadIdx.x;
  int l_idx = threadIdx.x+RADIUS; 

  // read input elements into shared memory
  temp[l_idx] = d_in[g_idx];
  if(threadIdx.x < RADIUS) {
    temp[l_idx - RADIUS] = d_in[g_idx - RADIUS];
    temp[l_idx + blockDim.x] = d_in[g_idx + blockDim.x];
   }

   //barrier
   __syncthreads(); 

   if (g_idx < n) {
      // apply the stencil
      int  result = 0;
      for(int  i = -RADIUS; i <= RADIUS; i++)
         result += temp[l_idx +RADIUS];
      // store the result to global memory 
      d_out[g_idx] = result;
   }
}

int main()
{

  // define the number of elements in the output array
  int n = 1<<11;

  // initialize input array & allocate output array on host
  int h_in[n + (2 * RADIUS)];
  int h_out[n];
  // initialize data on host
  for(int i = 0; i < (n + (2*RADIUS)); i++)
    h_in[i] = 1; // with a value of 1 and RADIUS of 3, all output values should be 7

  // declare pointers to device arrays
  int *d_in = 0;
  int *d_out = 0;

  // allocate memory on the device
  cudaMalloc((void**) &d_in, (n + (2 * RADIUS)) * sizeof(int));
  cudaMalloc((void**) &d_out, n * sizeof(int));

  // copy input data to device meory
  cudaMemcpy(d_in, h_in, (n + (2* RADIUS)) * sizeof(int), cudaMemcpyHostToDevice);

  // compute grid size
  size_t grid_size = n / BLOCK_SIZE;
  // deal with a possible partial final block
  if(n % BLOCK_SIZE) 
    ++grid_size;


  // define CUDA timers
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start CUDA timers
  cudaEventRecord(start, 0);


  // launch the kernel
  fast_stencil<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, n);

  // stop CUDA timers & sync stop time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // compute elpased time
  cudaEventElapsedTime(&time, start, stop);


  // copy the result back to the host memory
  cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

  // verify that every output value is 7
  int i;
  for(i = 0; i < n; ++i ) {
    if (h_out[i] != 7)
    {
      printf("Element at %d : %d != 7\n", i, h_out[i]);
      break;
    }
  }
  if (i == n)
    printf("SUCCESS!\n");

  //report GPU computing time
  printf ("Time for the kernel: %f ms\n", time);

  // deallocate device memory
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}

