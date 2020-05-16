#include <iostream>
#include <string>
#include <time.h>

#include <cudaErrors.h> 

#define ASCENDING true
#define DECENDING false

#define THEAD_MAX 1024
#define WARP_SIZE 32

__device__ void exchange(int i,int j,int* arr){
	int temp = arr[i];
	arr[i] = arr[j];
	arr[j] = temp;
}

__device__ void cmp(int i,int j,int* arr,bool direction) {
	if((arr[i] > arr[j]) == direction)
		exchange(i,j,arr);
}

__device__ int power2lessthan(int n){
	int k=1;
	while( k>0 && k < n ){
		k<<=1;
	}

	return k >> 1;
}

#ifdef UNIT_TEST
	#define n 4096
#else
	#define n 4096*32 // 2**17
#endif

__global__ void bitonicSortStep(int* arr,int step){

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int BX = blockDim.x;

	int i = bx*BX+tx;

	int direction = (1-(i/step)%2);

	for(int s=step;s>=1;s>>=1){
		__syncthreads();
		int m = i%s;
		int p = (i/s);
		int start = p*s*2 + m;
		cmp(start,start+s,arr,direction);
#ifdef UNIT_TEST
		if(i < n/2)
			printf("%d - %d %d - %d %d - %d\n",i,start,start+s,arr[start],arr[start+s],direction);
#endif

	}
}

int main() {
	srand (time(NULL));

	int max_threads = THEAD_MAX;
	int max_blocks = n/max_threads+1;

	dim3 blocks(max_blocks);
	dim3 threads(max_threads);

	int *arr;
	checkCudaErrors(cudaMallocManaged((void **)&arr, n*sizeof(*arr)));

	for(int i=0;i<n;i++) arr[i] = rand() % n;

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

	for(int i=1;i<n;i<<=1){
#ifdef UNIT_TEST
		std::cout << "Step :"<< i << std::endl;
#endif
		bitonicSortStep<<<blocks,threads>>>(arr,i);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
#ifdef UNIT_TEST
	//Expect everything to be n
		std::cout << "----------------------------" << std::endl;
		for(int k=0;k<n;k++){
			std::cout << arr[k] << " ";
		}
		std::cout<<std::endl;
#endif

	}

#ifdef UNIT_TEST
	//Expect everything to be n
		std::cout << "----------------------------" << std::endl;
		for(int k=0;k<n;k++){
			std::cout << arr[k] << " ";
		}
		std::cout<<std::endl;
#endif

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Time for gpu_version: "<<timer_seconds << " seconds"  << std::endl;


	checkCudaErrors(cudaFree(arr));

	return 0;
}