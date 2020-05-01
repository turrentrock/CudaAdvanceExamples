#include <stdio.h>
#include <iostream>
#include <time.h>
#include <iomanip>
#include <cmath>
#include <assert.h>

#include <cstdlib>
#include <ctime>

#include <cudaErrors.h>

template <typename T>
struct mat {
	T *buff;
	int rows;
	int cols;

	__host__ __device__ T& operator [] (int idx) { 
		return buff[idx]; 
	}

	__host__ __device__ T& at(int c,int r) { 
		if (r>=rows || c>=cols){
			printf("Bad Matrix Access %d %d\n",r,c);
			assert(0);
		}
		return buff[c*rows+r];
	}
};

__global__ void Init(mat<int> t1,mat<int> t2){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;

	if (i>=t1.rows || j >= t1.cols) return;

	t1.at(j,i) = j*t1.cols + i;
	t2.at(j,i) = 0;

	//printf("%d %d - %d %d\n",i,j,t1.at(j,i),t2.at(j,i));

}

__global__ void Transpose_rowRead_colWrite(mat<int> t1,mat<int> t2){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;

	if (i>=t1.rows || j >= t1.cols) return;

	t2.at(i,j) = t1.at(j,i);
}

__global__ void Transpose_colRead_rowWrite(mat<int> t1,mat<int> t2){
	int j = blockIdx.x*blockDim.x+threadIdx.x;
	int i = blockIdx.y*blockDim.y+threadIdx.y;

	if (i>=t1.rows || j >= t1.cols) return;

	t2.at(i,j) = t1.at(j,i);
}

int main(int argc,char** argv) {

	/* Matrix shape nx*ny */

	int n = 2048;

	int nx=n;
	int ny=n;

	/* Max tx * ty on my device */
	int tx=32;
	int ty=32;

	dim3 blocks(nx/tx+1,ny/ty+1);
	dim3 threads(tx,ty);

	mat<int> t1;
	mat<int> t2;

	t1.rows = t2.rows = nx;
	t1.cols = t2.cols = ny;
	checkCudaErrors(cudaMallocManaged((void **)&t1.buff, t1.rows*t1.cols*sizeof(*t1.buff)));
	checkCudaErrors(cudaMallocManaged((void **)&t2.buff, t2.rows*t2.cols*sizeof(*t2.buff)));

	Init<<<blocks,threads>>>(t1,t2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if (argc < 2) {
		std::cout << "Not enough args" <<std::endl;
		goto cleanup;
	}

	switch(argv[1][0]) {
		case '0' : 
			Transpose_rowRead_colWrite<<<blocks,threads>>>(t1,t2);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
			break;
		case '1' :
			Transpose_colRead_rowWrite<<<blocks,threads>>>(t1,t2);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
			break;
		default :
			std::cout << "Check the option entered : " << argv[1] << std::endl;
			break;
	}

#ifdef UNIT_TEST

	std::cout << "----------------------------------" << std::endl;
	for(int i=0;i<t1.cols;i++){
		for(int j=0;j<t1.rows;j++){
			std::cout << t1.at(i,j) << " ";
		}
		std::cout << std::endl;
	}

	std::cout << "----------------------------------" << std::endl;
	for(int i=0;i<t2.cols;i++){
		for(int j=0;j<t2.rows;j++){
			std::cout << t2.at(i,j) << " ";
		}
		std::cout << std::endl;
	}
#endif

cleanup:

	checkCudaErrors(cudaFree(t1.buff));
	checkCudaErrors(cudaFree(t2.buff));

	return 0;
}