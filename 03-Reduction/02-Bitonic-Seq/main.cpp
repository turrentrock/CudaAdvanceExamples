#include <iostream>
#include <cmath>
#include <time.h>

#define ASCENDING true
#define DECENDING false

void exchange(int i,int j,int* arr){
	int temp = arr[i];
	arr[i] = arr[j];
	arr[j] = temp;
}

void cmp(int i,int j,int* arr,bool direction) {
	if((arr[i] > arr[j]) == direction)
		exchange(i,j,arr);
}

void bitonicMerge(int lo,int n,int* arr,bool direction){
	if (n <= 1)
		return; 
	int mid = n/2; // Need to change this to make it work for non powers of two
	int k=lo;
	for(;k < lo+mid ;k++)
		cmp(k,k+mid,arr,direction);
	bitonicMerge(lo,mid,arr,direction);
	bitonicMerge(lo+mid,n-mid,arr,direction);
}

void bitonicSort(int lo,int n,int* arr,bool direction) {
	if (n <= 1)
		return;
#ifdef UNIT_TEST
	std::cout << "Lo:"<< lo << " N:"<< n <<std::endl;
#endif
	int mid = n/2;
	bitonicSort(lo,mid,arr,ASCENDING);
	bitonicSort(lo+mid,n-mid,arr,DECENDING);

	bitonicMerge(lo,n,arr,direction);
#ifdef UNIT_TEST
	for(int k=lo;k<lo+n;k++) {
		std::cout << arr[k]<<" ";
	}
	std::cout << std::endl;
#endif
}


#ifdef UNIT_TEST
	#define n 2048
#else
	#define n 67108864 // 2**26
#endif

int main() {

	int* arr;
	arr = new int[n];
	if(!arr)
		return 1;
	for(int i=0;i<n;i++) arr[i] = n-i;

	clock_t start,stop;
	start = clock();
	bitonicSort(0,n,arr,ASCENDING);
	stop = clock();

#ifdef UNIT_TEST
	for(int i=0;i<n;i++) std::cout << arr[i] << " ";
	std::cout << std::endl;
#endif
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Time for cpu_version: "<<timer_seconds << " seconds"  << std::endl;

	delete arr;
}