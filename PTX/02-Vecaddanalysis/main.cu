#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaErrors.h>

#define THREAD_MAX 1024

__global__ void Kernel(int* a,int* b,int *c,int n){

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ extern int shared_mem[];
	int reg;

	if(i>= n) return;

	reg = a[i] + b[i];
	shared_mem[i] = reg;
	c[i] = shared_mem[i];

}

int main(){
	int n=2048;

	int max_threads = THREAD_MAX;
	int max_blocks = ceil(n/max_threads);

	dim3 blocks(max_blocks);
	dim3 threads(max_threads);

	int *a,*b,*c;
	checkCudaErrors(cudaMalloc((void **)&a, n*sizeof(*a)));
	checkCudaErrors(cudaMalloc((void **)&b, n*sizeof(*b)));
	checkCudaErrors(cudaMalloc((void **)&c, n*sizeof(*c)));

	Kernel<<<blocks,threads,n*sizeof(*c)>>>(a,b,c,n);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	checkCudaErrors(cudaFree(a));
	checkCudaErrors(cudaFree(b));
	checkCudaErrors(cudaFree(c));

	return 0;
}
