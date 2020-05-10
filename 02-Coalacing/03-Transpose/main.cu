#include <iostream>
#include <string>

#include <cudaErrors.h>

__global__ void transpose_v0(float* a,float* b, int n){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i >= n || j >= n) return;

	b[n*i+j] = a[n*j+i];

}

__global__ void transpose_v1(float* a,float* b, int n){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i >= n || j >= n) return;

	b[n*j+i] = a[n*i+j];

}

int main(int argc,char** argv){

	int n=4096;
#ifdef UNIT_TEST
	n = 3;
#endif

	int max_thread_per_axis=32; // max_threads per block is 1024 and sqrt(1024) = 32 
	int max_mem = n*n;

	dim3 blocks(n/max_thread_per_axis+1,n/max_thread_per_axis+1);
	dim3 threads(max_thread_per_axis,max_thread_per_axis);

	float *a;
	checkCudaErrors(cudaMallocManaged((void **)&a, max_mem*sizeof(*a)));
	float *b;
	checkCudaErrors(cudaMallocManaged((void **)&b, max_mem*sizeof(*b)));

	for(int i=0;i<max_mem;i++) {a[i]=i;b[i]=0;}

	switch(argv[1][0]){
	case '0':
		transpose_v0<<<blocks,threads>>>(a,b,n);
		break;
	case '1':
		transpose_v1<<<blocks,threads>>>(a,b,n);
		break;
	}
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef UNIT_TEST
	//Expect everything to be n
	std::cout << "----------------------------" << std::endl;
	for(int j=0;j<n;j++){
		for(int i=0;i<n;i++){
			std::cout << b[j*n+i] << " ";
		}
		std::cout<<std::endl;
	}
#endif

	checkCudaErrors(cudaFree(a));
	checkCudaErrors(cudaFree(b));

	return 0;
}