// Sample codes for tutorial 4
#include <stdlib.h>
#include <stdio.h>

// the kernel computes reduce opertation on global memory
__global__ void global_reduce_kernel(float *d_out, float *d_in)
{
    // mapping threads to data on global memory
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;

    // iterate log n step; initialize s (stride) = 1; double s at each step  
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {   
        // All threads with IDs < s are active
        if (threadIdx.x < s)
        {
            //each thread adds value stride elements away to its own value
            d_in[global_idx] += d_in[global_idx + s];
        }
        // barrier to make sure all adds at one stage are done!
        __syncthreads();        
    }
    // only thread 0 writes result for this block back to global memory
    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = d_in[global_idx];
    }
}

// the kernel computes reduce opertation on shared memory
__global__ void shmem_reduce_kernel(float *d_out, const float *d_in)
{
    // shared data on shared memory allocated at the kernel call: 3rd arg to <<<b, t, shmem>>>
    __shared__ float sdata[];

    // mapping threads to data on global memory
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;

    // load data from global memory into shared memory
    sdata[threadIdx.x] = d_in[global_idx];
    // barrier to make sure entire block is loaded!
    __syncthreads();

    // iterate log n step; initialize s (stride) = 1; double s at each step 
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        // All threads with IDs < s are active
        if (threadIdx.x < s)
        {
            //each thread adds value stride elements away to its own value
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        // barrier to make sure all adds at one stage are done!
        __syncthreads();        
    }
    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

void reduce(float * d_out, float * d_intermediate, float * d_in, 
            int size, bool usesSharedMemory)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = size / maxThreadsPerBlock;
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
            (d_intermediate, d_in);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
            (d_intermediate, d_in);
    }
    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
            (d_out, d_intermediate);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
            (d_out, d_intermediate);
    }
}

int main(int argc, char **argv)
{
    const int ARRAY_SIZE = 1 << 20;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate input array on host
    float h_in[ARRAY_SIZE];
    float sum = 0.0f;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [-1.0f, 1.0f]
        h_in[i] = -1.0f + (float)random()/((float)RAND_MAX/2.0f);
        sum += h_in[i];
    }

   // declare pointers to device arrays
    float *d_in, *d_intermediate, *d_out;

    // allocate memory on the device
    cudaMalloc((void**) &d_in, ARRAY_BYTES);
    cudaMalloc((void**) &d_intermediate, ARRAY_BYTES); // overallocated
    cudaMalloc((void**) &d_out, sizeof(float));

    //copy input data to device memory
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    int whichKernel = 0;
    if (argc == 2) {
        whichKernel = atoi(argv[1]);
    }
        
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // launch the kernel
    switch(whichKernel) {
    case 0:
        printf("Running reduce operation with global memory\n");
        cudaEventRecord(start, 0);
        for (int i = 0; i < 100; i++)
        {
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
        }
        cudaEventRecord(stop, 0);
        break;
    case 1:
        printf("Running reduce operation with shared memory\n");
        cudaEventRecord(start, 0);
        for (int i = 0; i < 100; i++)
        {
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
        }
        cudaEventRecord(stop, 0);
        break;
    default:
        fprintf(stderr, "Error: ran no kernel\n");
        exit(EXIT_FAILURE);
    }
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime /= 100.0f;      // 100 trials

    // copy result to global memory
    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // check whether the result
    printf("The sume is %f \n", i, j, h_z[i][j]);

    //report GPU computing time
    printf("Average time for the kernel (100 traisl): %f ms\n", elapsedTime);

    //free device memory
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
        
    return 0;
}