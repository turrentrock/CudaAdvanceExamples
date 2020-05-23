#include <stdio.h>
#include <iostream>
#include <math.h>

// Handy Macro //
////////////////////////////////////////////////////////
#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)
////////////////////////////////////////////////////////

using namespace std;

__global__ void foo() {
	for(int i=0;i<1000;i++)
		pow(2,32);
}

int main() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Max Thread Dimensions: %i x %i x %i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("Max Block Dimensions: %i x %i x %i\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("Warp Size: %i\n", prop.warpSize);
	printf("Max Threads Per MultiProcessor: %i\n", prop.maxThreadsPerMultiProcessor);
	printf("MultiProcessor Count: %i\n", prop.multiProcessorCount);

	dim3 n_t(prop.maxThreadsDim[0]);
	dim3 n_b(prop.maxGridSize[0]);

	int *x;

	checkCudaErrors(cudaMalloc((void**)&x,pow(2,30)*sizeof(int)));;

	// Call Kernel //
	cudaError_t err;
	foo<<< n_b,n_t >>>();
	err = cudaGetLastError();
	if ( err != cudaSuccess ) {
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}

	// Sync all threads //
	checkCudaErrors(cudaDeviceSynchronize());

	return 0;
}