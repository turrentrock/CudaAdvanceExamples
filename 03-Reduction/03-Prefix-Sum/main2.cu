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

__global__ void prefixSumForward(float* arr,int step){

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int BX = blockDim.x;

	int i = bx*BX+tx;

	int ii = i+1;

	if( ii <= n &&  ii > n/float(step)) return;

	arr[ii*step-1] += arr[ii*step-step/2-1];

	if(step==n && n-1 == ii*step-1) {
		arr[ii*step]  = arr[ii*step-1];
		arr[ii*step-1]= 0;
	}
}

__global__ void prefixSumBackward(float* arr,int step){

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int BX = blockDim.x;

	int i = bx*BX+tx;

	int ii = i+1;

	if(i >= n || ii > n/float(step)) return;

	int temp = arr[ii*step-1];
	arr[ii*step-1]	 += arr[ii*step-step/2-1];
	arr[ii*step-step/2-1] = temp;
	
}

int main() {
	int max_threads = THEAD_MAX;

	float *arr;
	float *d_arr;
	arr = new float[n+1];

	checkCudaErrors(cudaMalloc((void **)&d_arr, (n+1)*sizeof(*arr)));

	for(int i=0;i<n;i++) arr[i] = i+1; arr[n] = 0;
	checkCudaErrors(cudaMemcpy(d_arr,arr,(n+1)*sizeof(*arr),cudaMemcpyHostToDevice));

#ifdef UNIT_TEST
	//Expect everything to be n
		std::cout << "----------------------------" << std::endl;
		for(int k=0;k<n+1;k++){
			std::cout << arr[k] << " ";
		}
		std::cout<<std::endl;
#endif

	clock_t start,stop;
	start = clock();

	for(int i=2;i<=n;i<<=1){

		int max_blocks = ceil(n/float(max_threads)/float(i));

		dim3 blocks(max_blocks);
		dim3 threads(max_threads);

		prefixSumForward<<<blocks,threads>>>(d_arr,i);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

	}

	for(int i=n;i>=2;i>>=1){

		int max_blocks = ceil(n/float(max_threads)/float(i));

		dim3 blocks(max_blocks);
		dim3 threads(max_threads);

		prefixSumBackward<<<blocks,threads>>>(d_arr,i);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

	}

	stop = clock();

	checkCudaErrors(cudaMemcpy(arr,d_arr,(n+1)*sizeof(*arr),cudaMemcpyDeviceToHost));
#ifdef UNIT_TEST
	//Expect everything to be n
		std::cout << "----------------------------" << std::endl;
		for(int k=0;k<n+1;k++){
			std::cout << arr[k] << " ";
		}
		std::cout<<std::endl;
#endif

	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Time for gpu_version_v2: "<<timer_seconds << " seconds"  << std::endl;


	checkCudaErrors(cudaFree(d_arr));
	delete arr;
	return 0;
}