#include <iostream>
#include <time.h>


#ifdef UNIT_TEST
	#define n 4096
#else
	#define n 67108864 // 2**26 
#endif


void prefixSum(int* arr) {

	for(int i=0;i<n-1;i++){
		arr[i+1] += arr[i];
	}

}

int main() {

	int *arr = new int[n];

	for(int i=0;i<n;i++) arr[i] = i + 1;

#ifdef UNIT_TEST
	//Expect everything to be n
		std::cout << "----------------------------" << std::endl;
		for(int k=0;k<n;k++){
			std::cout << arr[k] << " ";
		}
		std::cout<<std::endl;
#endif

	clock_t start,stop;
	start = clock();

	prefixSum(arr);

	stop = clock();

#ifdef UNIT_TEST
	//Expect everything to be n
		std::cout << "----------------------------" << std::endl;
		for(int k=0;k<n;k++){
			std::cout << arr[k] << " ";
		}
		std::cout<<std::endl;
#endif

	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Time for cpu_version: "<<timer_seconds << " seconds"  << std::endl;

	delete arr;

	return 0;
}