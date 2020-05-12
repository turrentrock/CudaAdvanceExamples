#include <iostream>
#include <string>
#include <time.h>

#include <cudaErrors.h> 

#define THEAD_MAX 1024

__global__ void reduce_v0(float* in,float* out, int n){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int BX = blockDim.x; //same as THEAD_MAX
	int i  = bx*BX+tx;

	__shared__ float S[THEAD_MAX];

	S[tx] = i < n ?  in[i] : 0;
	__syncthreads();
	for(int s=1; s<BX ;s*=2){
		if(tx%(2*s)==0)
			S[tx] += S[tx+s];
		__syncthreads();
	}
	if(tx==0)
		out[bx] = S[0];
}


__global__ void reduce_v1(float* in,float* out, int n){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int BX = blockDim.x; //same as THEAD_MAX
	int i  = bx*BX+tx;

	__shared__ float S[THEAD_MAX];

	S[tx] = i < n ?  in[i] : 0;
	__syncthreads();
	for(int s=1; s<BX ;s*=2){
		int index = 2*s*tx;
		if(index < BX)
			S[index] += S[index+s];
		__syncthreads();
	}
	if(tx==0)
		out[bx] = S[0];
}



int main(int argc,char** argv){

	int n=16384;
#ifdef UNIT_TEST
	n = 1025;
#endif

	int max_threads = THEAD_MAX;
	int max_blocks = n/max_threads+1;

	dim3 blocks(max_blocks);
	dim3 threads(max_threads);

	float *in;
	float *out;
	checkCudaErrors(cudaMallocManaged((void **)&in, n*sizeof(*in)));
	checkCudaErrors(cudaMallocManaged((void **)&out, max_blocks*sizeof(*out)));

	for(int i=0;i<n;i++) {in[i]=1;}

	clock_t start,stop;
	start = clock();
	switch(argv[1][0]){
	case '0':
		reduce_v0<<<blocks,threads>>>(in,out,n);
		break;
	case '1':
		reduce_v1<<<blocks,threads>>>(in,out,n);
		break;
	}
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Time for v"<<argv[1][0]<<": "<<timer_seconds << " seconds"  << std::endl;

#ifdef UNIT_TEST
	//Expect everything to be n
	std::cout << "----------------------------" << std::endl;
	for(int i=0;i<max_blocks;i++){
		std::cout << out[i] << " ";
	}
	std::cout<<std::endl;
#endif

	checkCudaErrors(cudaFree(in));
	checkCudaErrors(cudaFree(out));

	return 0;
}