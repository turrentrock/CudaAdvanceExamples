#include <iostream>
#include <cublas_v2.h>
#include "cudaErrors.h"

// HM1 -> [NxK] <- CM1
// HM2 -> [KxM] <- CM2
// HM3 -> [NxM] <- CM3

#define N 3
#define M 3
#define K 2 

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

using namespace std;

void printFloatMat(float* mat,int n,int m) {
	for(int i=0;i<n;i++) {
		for(int j=0;j<m;j++)
			cout << mat[IDX2C(i,j,n)] << " ";
		cout << endl;
	}
}

int main() {

	//  CUBLAS  Context //
	cublasHandle_t handle;
	checkCublasErrors(cublasCreate(&handle));

	// Init Matricies //
	float* HM1 = (float*)malloc(N*K*sizeof(float));
	float* HM2 = (float*)malloc(K*M*sizeof(float));
	float* HM3 = (float*)malloc(N*M*sizeof(float));
	for(int i=0;i<N*K || i<K*M || i< N*M ;i++) {
		if (i<N*K)
			HM1[i] = i;
		if (i<K*M)
			HM2[i] = i;
		if (i<N*M)
			HM3[i] = i;
	}

	// HM prints //
	cout << "-------------------------" << endl ;
	printFloatMat(HM1,N,K);
	cout << "-------------------------" << endl ;
	printFloatMat(HM2,K,M);
	cout << "-------------------------" << endl ;
	printFloatMat(HM3,N,M);
	cout << "-------------------------" << endl ;

	// Init CM //
	float *CM1,*CM2,*CM3;
	checkCudaErrors(cudaMalloc(&CM1,N*K*sizeof(float)));
	checkCudaErrors(cudaMalloc(&CM2,K*M*sizeof(float)));
	checkCudaErrors(cudaMalloc(&CM3,N*M*sizeof(float)));
	checkCublasErrors(cublasSetMatrix(N,K,sizeof(*HM1),HM1,N,CM1,N));
	checkCublasErrors(cublasSetMatrix(K,M,sizeof(*HM2),HM2,K,CM2,K));
	checkCublasErrors(cublasSetMatrix(N,M,sizeof(*HM3),HM3,N,CM3,N));
/*	//Also Valid Copy
	checkCudaErrors(cudaMemcpy(CM1,HM1,N*K*sizeof(float),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(CM2,HM2,K*M*sizeof(float),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(CM3,HM3,N*M*sizeof(float),cudaMemcpyHostToDevice));
*/

// 1. Scale Example
	// Scale Factor //
	float alpha = 0.1;
	// Apply Alpha every //
	int stride = 1;
	// CM3 = alpha * CM3 //
	checkCublasErrors(cublasSscal(handle,N*M,&alpha,CM3,stride));
	// Copy to Host //
	checkCublasErrors(cublasGetMatrix(N,M,sizeof(*HM3),CM3,N,HM3,N));
	// Output //
	cout << "After Scale HM3" << endl;
	printFloatMat(HM3,N,M);
//

// 2. Matmult
	// Scale Factors //
	float al = 1;
	float bt = 1;
	// When CUBLAS_OP_N,CUBLAS_OP_N : CM3 = al*CM1*CM2 + bt*CM3 //
	checkCublasErrors(cublasSgemm(handle,
								CUBLAS_OP_N,CUBLAS_OP_N,
								N,M,K,
								&al,
								CM1,N,
								CM2,K,
								&bt,
								CM3,N
								));
	// Copy to Host //
	checkCublasErrors(cublasGetMatrix(N,M,sizeof(*HM3),CM3,N,HM3,N));
	// Output //
	cout << "After HM3=HM1*HM2+HM3" << endl;
	printFloatMat(HM3,N,M);
//

	// Free //
	cudaFree(CM1);
	cudaFree(CM2);
	cudaFree(CM3);
	free(HM1);
	free(HM2);
	free(HM3);
	return 0;
}