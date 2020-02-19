#include <iostream>
#include <vector>
#include <tuple>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "getImageChannels.h"

using namespace std;
using namespace cv;
using namespace cuda;

////////////////////////////////////////////////////////
#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)
////////////////////////////////////////////////////////

__global__ void blur(int* B,int* G,int* R, 
					 int* RB,int* RG,int* RR,
					 int* K, int rows, int cols, 
					 int krows, int kcols) {

	int index = blockIdx.x * 1024 + threadIdx.x;

	if (index > rows*cols)
		return;

	int pixel_row = index/cols ; 
	int pixel_col = index - pixel_row*cols;

	int pr,pc,idx;

	int k_sum = 0;
	int kr,kc;

	int k_center_row = (krows-1)/2;
	int k_center_col = (kcols-1)/2;

	//printf("%d %d\n", pixel_row , pixel_col);

	for(int i=0;i<krows;i++) {
		for(int j=0;j<kcols;j++) {
			
			kr = (i - k_center_row);
			kc = (j - k_center_col);

			pr = pixel_row + kr ; 
			pc = pixel_col + kc ;

			idx = pr*cols + pc;

			if (pr >=0 && pr < rows && pc>=0 && pc < cols) {
				k_sum += K[kr*kcols + kc];
				
				RB[index] += B[idx]*K[kr*kcols + kc];
				RG[index] += G[idx]*K[kr*kcols + kc];
				RR[index] += R[idx]*K[kr*kcols + kc];

			}
		}
	}

	RB[index] /= k_sum;
	RG[index] /= k_sum;
	RR[index] /= k_sum;
}

int* cudaMallocIntMatrix(int n,int m) {
	// First Allocate Pointers to Arrays //
	int *CM;
	checkCudaErrors(cudaMalloc((void**)&CM, n*m*sizeof(int)));
	return CM;
}

int main(int argc, char** argv ) {
	// Input for image as arg //
	if( argc != 2)
	{
		cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	// Image Load //
	vector<int*> channels;
	int rows,cols;
	tie(channels,rows,cols) = getIntArrays(argv[1]);

	// Allocate channels //
	int *B,*G,*R,*RB,*RG,*RR,*K;
	int KH[] = {1,1,1,1,1,
			  	1,1,1,1,1,
			  	1,1,1,1,1,
			  	1,1,1,1,1,
			    1,1,1,1,1};
	int k_r=5;
	int k_c=5;

	B = cudaMallocIntMatrix(rows,cols);
	G = cudaMallocIntMatrix(rows,cols);
	R = cudaMallocIntMatrix(rows,cols);

	RB = cudaMallocIntMatrix(rows,cols);
	RG = cudaMallocIntMatrix(rows,cols);
	RR = cudaMallocIntMatrix(rows,cols);

	K  = cudaMallocIntMatrix(k_r,k_c);

	// Copy Channels to Device //
	checkCudaErrors(cudaMemcpy(B,channels[0],rows*cols*sizeof(int),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(G,channels[1],rows*cols*sizeof(int),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(R,channels[2],rows*cols*sizeof(int),cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(K,KH,k_r*k_c*sizeof(int),cudaMemcpyHostToDevice));

	// Grid Shape //
	int n_b = rows*cols / 1024 + 1;
	int n_t = 1024;

	dim3 thread_dim(n_t);
	dim3 block_dim(n_b);

	cout << "No. Blocks:" << n_b << " No. Thread Per Block:" << n_t << endl;
	cout << "Rows :" << rows << " Cols :" << cols << endl;

	// Call Kernel //
	cudaError_t err;
	blur<<< block_dim,thread_dim >>>(B,G,R,
									 RB,RG,RR,
									 K,
									 rows,cols,
									 k_r,k_c);
	err = cudaGetLastError();
	if ( err != cudaSuccess ) {
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}

	// 	Calling Kernels to Run //
	checkCudaErrors(cudaDeviceSynchronize());

	// Copy image back to guest //
	checkCudaErrors(cudaMemcpy(channels[0],RB,rows*cols*sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(channels[1],RG,rows*cols*sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(channels[2],RR,rows*cols*sizeof(int),cudaMemcpyDeviceToHost));

	// Save to img //
	saveImage(channels,rows,cols);

	// Free Cuda Mem //
	cudaFree(B);
	cudaFree(G);
	cudaFree(R);

	// Free vector mem //
	for(int i=0;i<3;i++)
		free(channels[i]);

	return 0;
}