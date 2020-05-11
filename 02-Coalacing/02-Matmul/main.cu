#include <iostream>
#include <string>

#include <cudaErrors.h>

__global__ void matmul_v0(float* a,float* b,float* c, int n){
	// C(nxn) = A(nxn) * B(nxn);
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i >= n || j >= n) return;

	float c_ij = 0;
	for(int k=0;k<n;k++){
		c_ij += a[n*j+k]*b[n*k+i];

//		printf("%d %d %d : %f %f\n",i,j,k,a[n*j+k],b[n*k+i]);

	}
	c[n*j+i] = c_ij;

}

#define TILE_SIZE 32

__global__ void matmul_v1(float* a,float* b,float* c, int n){
	// C(nxn) = A(nxn) * B(nxn);

	__shared__ float A[TILE_SIZE][TILE_SIZE+1];
	__shared__ float B[TILE_SIZE][TILE_SIZE+1];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int i = bx*TILE_SIZE+tx;
	int j = by*TILE_SIZE+ty;

	A[ty][tx] = A[ty][tx] = 0;
	if(i >= n || j >= n) return;

	float c_ij = 0;
	for(int m=0;m<float(n)/TILE_SIZE;m++){
		A[ty][tx] = a[j*n+ m*TILE_SIZE + tx];
		B[ty][tx] = b[(m*TILE_SIZE+ty)*n+i];

//		printf("%d %d : %f - %f\n",tx,ty,A[ty][tx],B[ty][tx]);

		__syncthreads();

		for(int k=0;k<TILE_SIZE;k++)
			c_ij += A[ty][k]*B[k][tx];
		__syncthreads();
	}
	c[n*j+i] = c_ij;

}

int main(int argc,char** argv){

	int n=16384;
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
	float *c;
	checkCudaErrors(cudaMallocManaged((void **)&c, max_mem*sizeof(*c)));

	for(int i=0;i<max_mem;i++) {a[i]=1;b[i]=1;c[i]=0;}

	switch(argv[1][0]){
	case '0':
		matmul_v0<<<blocks,threads>>>(a,b,c,n);
		break;
	case '1':
		matmul_v1<<<blocks,threads>>>(a,b,c,n);
		break;
	}
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef UNIT_TEST
	//Expect everything to be n
	std::cout << "----------------------------" << std::endl;
	for(int j=0;j<n;j++){
		for(int i=0;i<n;i++){
			std::cout << c[j*n+i] << " ";
		}
		std::cout<<std::endl;
	}
#endif

	checkCudaErrors(cudaFree(a));
	checkCudaErrors(cudaFree(b));
	checkCudaErrors(cudaFree(c));

	return 0;
}