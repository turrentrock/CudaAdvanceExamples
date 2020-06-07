#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void Kernel(int* arr){

	asm( 
		"mov.s32 %2,10;"
		"mov.s32 %1,15;"
		"add.s32 %0,%1,%2;"
		: "=r"(arr[0]) , "=r"(arr[1])
		: "r"(arr[2])
	);

}

int main(){
	int i[3] = {
		0,1,2
	};

	int *d_i;
	cudaMalloc(&d_i,sizeof(int)*3);
	cudaMemcpy(d_i,i,sizeof(int)*3,cudaMemcpyHostToDevice);

	Kernel<<<1,1>>>(d_i);
	cudaDeviceSynchronize();

	cudaMemcpy(i,d_i,sizeof(int)*3,cudaMemcpyDeviceToHost);

	std::cout << "Result is "<< i[0] << " " 
	                         << i[1] << " "
	                         << i[2] << std::endl;

	cudaFree(d_i);

	return 0;
}
