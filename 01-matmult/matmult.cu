#include <stdio.h>
#define N 3
#define M 3

/************* Neet way to handle cuda errors ***********/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
/*********************************************************/

__global__ void mult(int* A,int* B,int* C) {
	int x = threadIdx.x;
	int y = threadIdx.y;

	if ( x >= N || y >= M )
		return;

	for(int i=0,j=0; i < N && j < M ; i++, j++) {
			C[x*N+y] += A[x*N+j]*B[i*N+y];
		}
}

int* cudaMallocIntMatrix(int n,int m) {
	// First Allocate Pointers to Arrays //
	int *CM;
	gpuErrchk(cudaMalloc(&CM, n*m*sizeof(int)))
	return CM;
}

int main() {
	// Dimentions of the Array NxM //
	dim3 NM(N,M);
	cudaError_t err;

	// Local Matrices A,B,C //
	int A[N*M],B[N*M],C[N*M];

	// Initializing Local A,B,C with some values//
	for(int i=0;i<N;i++) {
		for(int j=0;j<M;j++){
			A[i*N+j] = i + j;
			B[i*N+j] = i - j;
			C[i*N+j] = 0;
		}
	}

	// Input Matrices //
	printf("Matrix A\n");
	printf("--------------\n");
	for(int i=0;i<N;i++) {
		for(int j=0;j<M;j++){
			printf("%d ",A[i*N+j]);
		}
		printf("\n");
	}
	printf("--------------\n");

	printf("Matrix B\n");
	printf("--------------\n");
	for(int i=0;i<N;i++) {
		for(int j=0;j<M;j++){
			printf("%d ",B[i*N+j]);
		}
		printf("\n");
	}
	printf("--------------\n");

	// Cuda Memory Allocation //
	int *CA,*CB,*CC;
	CA = cudaMallocIntMatrix(N,M);
	CB = cudaMallocIntMatrix(N,M);
	CC = cudaMallocIntMatrix(N,M);

	// Copy to CPUt //
	gpuErrchk(cudaMemcpy(CA,A,N*M*sizeof(int),cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(CB,B,N*M*sizeof(int),cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(CC,C,N*M*sizeof(int),cudaMemcpyHostToDevice))

	// Init Kernel //
	mult<<<1,NM>>>(CA,CB,CC);
	err = cudaGetLastError();
	if ( err != cudaSuccess ) {
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}

	// 	Calling Kernels to Run //
	gpuErrchk(cudaDeviceSynchronize())

	// Copy to Host //
	gpuErrchk(cudaMemcpy(A,CA,N*M*sizeof(int),cudaMemcpyDeviceToHost))
	gpuErrchk(cudaMemcpy(B,CB,N*M*sizeof(int),cudaMemcpyDeviceToHost))
	gpuErrchk(cudaMemcpy(C,CC,N*M*sizeof(int),cudaMemcpyDeviceToHost))

	// Output//
	printf("Matrix C\n");
	printf("--------------\n");
	for(int i=0;i<N;i++) {
		for(int j=0;j<M;j++){
			printf("%d ",C[i*N+j]);
		}
		printf("\n");
	}
	printf("--------------\n");

	gpuErrchk(cudaFree(CA))
	gpuErrchk(cudaFree(CB))
	gpuErrchk(cudaFree(CC))

	return 0;
}