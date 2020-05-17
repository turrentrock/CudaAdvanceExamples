#include <iostream>
#include <string>
#include <time.h>

#include <cudaErrors.h> 

#define THEAD_MAX 1024
#define WARP_SIZE 32

#ifdef UNIT_TEST
	#define n 4096
#else
	#define n 67108864 // 2**26 
#endif

__global__ void prefixSum(float* arr,int step){

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int BX = blockDim.x;

	int i = bx*BX+tx;

	if(i < step) return;

	int temp = arr[i-step];
	__syncthreads();
	arr[i] += temp;
}

int main() {

	int max_threads = THEAD_MAX;
	int max_blocks = ceil(n/float(max_threads));

	dim3 blocks(max_blocks);
	dim3 threads(max_threads);

	float *arr;
	float *d_arr;
	arr = new float[n];

	checkCudaErrors(cudaMalloc((void **)&d_arr, n*sizeof(*arr)));

	for(int i=0;i<n;i++) arr[i] = i+1;
	checkCudaErrors(cudaMemcpy(d_arr,arr,n*sizeof(*arr),cudaMemcpyHostToDevice));

#ifdef UNIT_TEST
	//Expect everything to be n
		std::cout << "----------------------------" << std::endl;
		for(int k=0;k<n;k++){
			std::cout << arr[k] << " ";
		}
		std::cout<<std::endl;
#endif

	clock_t start,stop;
	start = clock();

	for(int i=1;i<=n;i<<=1){
		prefixSum<<<blocks,threads>>>(d_arr,i);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	stop = clock();

	checkCudaErrors(cudaMemcpy(arr,d_arr,n*sizeof(*arr),cudaMemcpyDeviceToHost));
#ifdef UNIT_TEST
	//Expect everything to be n
		std::cout << "----------------------------" << std::endl;
		for(int k=0;k<n;k++){
			std::cout << arr[k] << " ";
		}
		std::cout<<std::endl;
#endif

	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Time for gpu_version_v1: "<<timer_seconds << " seconds"  << std::endl;


	checkCudaErrors(cudaFree(d_arr));
	delete arr;
	return 0;
}