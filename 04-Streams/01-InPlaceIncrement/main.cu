#include <iostream>
#include <string>
#include <time.h>
#include <stdlib.h>

#include <cudaErrors.h>

#define THEAD_MAX 1024

__global__ void kernel(int* arr,int offset_min,int n){

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int BX = blockDim.x;

	int i = bx*BX+tx;

	if (i>= n|| i < 0) return;
	//printf("%d %d - %d %d\n",offset_min,offset_max,i+offset_min,i);
	arr[i+offset_min] += 1;

}

void display(int* arr,int n){
#ifdef UNIT_TEST
	std::cout << "----------------------------" << std::endl;
	for(int k=0;k<n;k++)
		std::cout << arr[k] << " ";
	std::cout<<std::endl;
#endif
}

int main(int argc,char** argv) {

#ifdef UNIT_TEST
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout << "Max concurrentKernels: " << prop.concurrentKernels <<std::endl;;
#endif

	int n = pow(2, 26);
#ifdef UNIT_TEST
	n = 4;
#endif
	int max_threads = THEAD_MAX;
	int max_blocks = ceil(n/float(max_threads));

	int num_streams = atoi(argv[1]);
	if (num_streams > n) return -1;

	cudaStream_t streams[num_streams];
	for(int i=0;i<num_streams;i++){
		checkCudaErrors(cudaStreamCreate(&streams[i]));
	}

	max_blocks = ceil(max_blocks/float(num_streams));

	dim3 blocks(max_blocks);
	dim3 threads(max_threads);

	int  *arr;
	int  *d_arr;
	
	int unit_size = ceil(n/float(num_streams));
	checkCudaErrors(cudaMallocHost((void **)&arr, n*sizeof(*arr))); //Pinned mem in ram
	checkCudaErrors(cudaMalloc((void **)&d_arr, n*sizeof(*d_arr)));

	for(int i=0;i<n;i++) arr[i] = i+1;

	display(arr,n);

	cudaEvent_t start,stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start));

	//Copy Data Aynchronusly in units of data.
	//Split Factor is determined using num_streams
	for(int i=0;i<num_streams-1;i++){
		checkCudaErrors(cudaMemcpyAsync(d_arr+i*unit_size,arr+i*unit_size,unit_size*sizeof(*arr),cudaMemcpyHostToDevice,streams[i]));
		kernel<<<blocks,threads,0,streams[i]>>>(d_arr,i*unit_size,unit_size);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaMemcpyAsync(arr+i*unit_size,d_arr+i*unit_size,unit_size*sizeof(*arr),cudaMemcpyDeviceToHost,streams[i]));
	}
	if(unit_size != n/num_streams){
		int smaller_unit_size = n - (num_streams-1)*unit_size;
		int i = num_streams-1;
		checkCudaErrors(cudaMemcpyAsync(d_arr+i*unit_size,arr+i*unit_size,smaller_unit_size*sizeof(*arr),cudaMemcpyHostToDevice,streams[i]));	
		kernel<<<blocks,threads,0,streams[i]>>>(d_arr,i*unit_size,smaller_unit_size);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaMemcpyAsync(arr+i*unit_size,d_arr+i*unit_size,smaller_unit_size*sizeof(*arr),cudaMemcpyDeviceToHost,streams[i]));
	} else {
		int i = num_streams-1;
		checkCudaErrors(cudaMemcpyAsync(d_arr+i*unit_size,arr+i*unit_size,unit_size*sizeof(*arr),cudaMemcpyHostToDevice,streams[i]));
		kernel<<<blocks,threads,0,streams[i]>>>(d_arr,i*unit_size,unit_size);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaMemcpyAsync(arr+i*unit_size,d_arr+i*unit_size,unit_size*sizeof(*arr),cudaMemcpyDeviceToHost,streams[i]));
	}
	
	checkCudaErrors(cudaEventRecord(stop));
	//checkCudaErrors(cudaEventSynchronize(stop));
	while(cudaEventQuery(stop) == cudaErrorNotReady);

	float milliseconds=0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time for split factor-"<<num_streams<<": "<<milliseconds/1000 << " seconds"  << std::endl;

	display(arr,n);

	checkCudaErrors(cudaFree(d_arr));
	checkCudaErrors(cudaFreeHost(arr));

	return 0;
}