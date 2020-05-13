#include <iostream>
#include <string>
#include <time.h>

#include <cudaErrors.h> 

#define THEAD_MAX 1024
#define WARP_SIZE 32

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

__global__ void reduce_v2(float* in,float* out, int n){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int BX = blockDim.x; //same as THEAD_MAX
	int i  = bx*BX+tx;

	__shared__ float S[THEAD_MAX];

	S[tx] = i < n ?  in[i] : 0;
	__syncthreads();
	for(int s=BX/2; s>0 ;s>>=1){
		if(tx < s)
			S[tx] += S[tx+s];
		__syncthreads();
	}
	if(tx==0)
		out[bx] = S[0];
}


__global__ void reduce_v3(float* in,float* out, int n){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int BX = blockDim.x; //same as THEAD_MAX
	int i  = bx*(BX*2)+tx;

	__shared__ float S[THEAD_MAX];

	S[tx] = in[i] + in[i+BX]; //Increased part thread activity at start and start only half the threads
	__syncthreads();
	for(int s=BX/2; s>0 ;s>>=1){
		if(tx < s)
			S[tx] += S[tx+s];
		__syncthreads();
	}
	if(tx==0)
		out[bx] = S[0];
}

__device__ void warp_reduce(float* S,int tx){
	S[tx] += S[tx + 32]; __syncthreads();
	S[tx] += S[tx + 16]; __syncthreads();
	S[tx] += S[tx + 8];  __syncthreads();
	S[tx] += S[tx + 4];  __syncthreads();
	S[tx] += S[tx + 2];  __syncthreads();
	S[tx] += S[tx + 1];  __syncthreads();
}

__global__ void reduce_v4(float* in,float* out, int n){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int BX = blockDim.x; //same as THEAD_MAX
	int i  = bx*(BX*2)+tx;

	__shared__ float S[THEAD_MAX];

	S[tx] = in[i] + in[i+BX]; //Increased part thread activity at start and start only half the threads
	__syncthreads();
	for(int s=BX/2; s>WARP_SIZE ;s>>=1){
		if(tx < s)
			S[tx] += S[tx+s];
		__syncthreads();
	}
	if(tx < WARP_SIZE)
		warp_reduce(S,tx);				//Unroaling the last warp
	if(tx==0)
		out[bx] = S[0];
}

template<unsigned int BX>
__device__ void warpReduce(float* S,int tx){
	if(BX >= 64) {S[tx] += S[tx + 32]; __syncthreads();}
	if(BX >= 32) {S[tx] += S[tx + 16]; __syncthreads();}
	if(BX >= 16) {S[tx] += S[tx + 8];  __syncthreads();}
	if(BX >=  8) {S[tx] += S[tx + 4];  __syncthreads();}
	if(BX >=  4) {S[tx] += S[tx + 2];  __syncthreads();}
	if(BX >=  2) {S[tx] += S[tx + 1];  __syncthreads();}
}

template<unsigned int BX>
__global__ void reduce_v5(float* in,float* out, int n){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int i  = bx*(BX*2)+tx;

	__shared__ float S[BX];	//Want to have only BX amount of shared mem which is THREAD_MAX in previous

	S[tx] = in[i] + in[i+BX]; //Increased part thread activity at start and start only half the threads
	__syncthreads();

	if(BX >= 1024){                 // Max threads for block in my gpu is 1024
		if(tx < 512)
			S[tx] += S[tx+512];
		__syncthreads();
	}

	if(BX >= 512){
		if(tx < 256)
			S[tx] += S[tx+256];
		__syncthreads();
	}

	if(BX >= 256){
		if(tx < 128)
			S[tx] += S[tx+128];
		__syncthreads();
	}

	if(BX >= 128){
		if(tx < 64)
			S[tx] += S[tx+64];
		__syncthreads();
	}

	if(tx < WARP_SIZE) {				//WARP_SIZE is 32 
		warp_reduce(S,tx);				//Unroaling the last warp
	}

	if(tx==0)
		out[bx] = S[0];
}


int main(int argc,char** argv){

	int n=67108864;
#ifdef UNIT_TEST
	n = 2049;
#endif

	int max_threads = THEAD_MAX;
	int max_blocks = n/max_threads+1;

	dim3 blocks(max_blocks);
	dim3 threads(max_threads);

	float *in;
	float *out;
	checkCudaErrors(cudaMallocManaged((void **)&in, n*sizeof(*in)));
	checkCudaErrors(cudaMallocManaged((void **)&out, max_blocks*sizeof(*out)));

	for(int i=0;i<n;i++) {in[i]=i;}

	clock_t start,stop;
	start = clock();
	switch(argv[1][0]){
	case '0':
		reduce_v0<<<blocks,threads>>>(in,out,n);
		break;
	case '1':
		reduce_v1<<<blocks,threads>>>(in,out,n);
		break;
	case '2':
		reduce_v2<<<blocks,threads>>>(in,out,n);
		break;
	case '3':
		{	dim3 blocks(n/max_threads/2+1);
			reduce_v3<<<blocks,threads>>>(in,out,n);
		}
		break;
	case '4':
		{
			dim3 blocks(n/max_threads/2+1);
			reduce_v4<<<blocks,threads>>>(in,out,n);
		}
		break;
	case '5':
		{
			dim3 blocks(n/max_threads/2+1);
			reduce_v5<THEAD_MAX><<<blocks,threads>>>(in,out,n);
		}
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