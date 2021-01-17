#include "Sort.h"
#include <cassert>
#include <cfloat>
#include <cuda_runtime.h>
#include "cuda_utilities.h"
#include "sharedmem.h"
#include "Functional.h"


namespace PhysIKA {
#define MAX_LENGTH 10000
	Sort::Sort() {

	}

	Sort::Sort(int length) {

	}

	Sort::~Sort() {
		cudaFree(num);
	}
	int * Sort::radixSort(int * arr, int length) {
		numMalloc(arr, length);
		int maxNum = INT_MIN;
		for (int i = 0; i < length; i++) {
			if (arr[i] > maxNum) {
				maxNum = arr[i];
			}
		}
		int bitLength = int2bit(maxNum);
		deviceRadixSort << <1, bitLength>> > (arr, length);
		cudaMemcpy((void*)arr, (void*)num, length * sizeof(int), cudaMemcpyDeviceToHost);
		return arr;
	}

	int Sort::int2bit(int n) {
		int count = 0;
		while (n != 0) {
			n = n / 2;
			count++;
		}
		return count;
	}

	__device__ int getBinaryByN(int num, int id) {
		int count = 0;
		int temp;
		while (num != 0) {
			temp = num % 2;
			num = num / 2;
			count++;
			if (count == id) {
				return temp;
			}
		}
		return 0;
	}

	__global__ void deviceRadixSort(int *arr, int length) {
		extern __shared__ int a0[MAX_LENGTH];
		extern __shared__ int a1[MAX_LENGTH];
		int id = threadIdx.x;
		int k0 = 0;
		int k1 = 0;
		for (int i = 0; i < length; i++) {
			int x = getBinaryByN(arr[i], id + 1);
			if (x == 0) {
				a0[k0] = arr[i];
				k0++;
			}else if (x == 1) {
				a1[k1] = arr[i];
				k1++;
			}
			__syncthreads();
		}
		for (int i = 0; i < k0; i++) {
			arr[i] = a0[i];
		}
		__syncthreads();
		for (int i = k0, i1 = 0; i < k0 + k1; i1++, i++) {
			arr[i] = a1[i1];
		}
		__syncthreads();
	}


	void Sort::numMalloc(int *arr, int length) {
		cudaMalloc((void**)&num, length * sizeof(int));
		cudaMemcpy((void*)num, (void*)arr, length * sizeof(int), cudaMemcpyHostToDevice);
	}
}