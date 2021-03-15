#include "SortCuda.h"
#include <cassert>
#include <cfloat>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <climits>
#include<cstdio>
#include<cstdlib>
#include<string>



namespace PhysIKA {
#define MAX_LENGTH 1024
	template<typename T>
	SortCuda<T>::SortCuda() {
		this->sortLength = 0;
	}

	template<typename T>
	SortCuda<T>::SortCuda(int length) {
		this->sortLength = length;
	}

	template<typename T>
	SortCuda<T>::~SortCuda() {
		free(this->host_input);
		free(this->host_res);
		cudaFree(this->device_input);
		cudaFree(this->device_res);
	}
/*
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
		__shared__ int a0[MAX_LENGTH];
		__shared__ int a1[MAX_LENGTH];
		int id = threadIdx.x;
		int k0 = 0;
		int k1 = 0;
		for (int i = 0; i < length; i++) {
			int x = getBinaryByN(arr[i], id + 1);
			if (x == 0) {
				a0[k0] = arr[i];
				k0++;
			}
			else if (x == 1) {
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
	void SortCuda::radixSort(int * arr, int length) {
		numMalloc(arr, length);
		int maxNum = INT_MIN;
		for (int i = 0; i < length; i++) {
			if (arr[i] > maxNum) {
				maxNum = arr[i];
			}
		}
		int bitLength = int2bit(maxNum);
		deviceRadixSort << <1, bitLength>> > (num, length);
		cudaMemcpy((void*)arr, (void*)num, length * sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(num);
	}
	*/

	template<typename T>
	__device__ inline void swap(T &num1, T &num2, int dir) {
		T temp;
		if ((num1 > num2) == dir) {
			temp = num1;
			num1 = num2;
			num2 = temp;
		}
	}

	template<typename T>
	__global__ void smallBinoticSort(T *arr, T *res, int length, int dir) {
		__shared__ T buf[MAX_LENGTH];
		buf[threadIdx.x] = arr[threadIdx.x];
		buf[threadIdx.x + (MAX_LENGTH / 2)] = arr[threadIdx.x + (MAX_LENGTH / 2)];
		//printf("%d\n", arr[threadIdx.x]);
		//printf("%d\n", arr[threadIdx.x+ (MAX_LENGTH / 2)]);
		__syncthreads();
		for (int size = 2; size < length; size<<=1) {
			int d = dir ^ ((threadIdx.x & (size / 2)) != 0);
			for (int j = size /2; j > 0; j >>= 1) {
				__syncthreads();
				int pos = 2 * threadIdx.x - (threadIdx.x&(j - 1));
				swap(buf[pos], buf[pos + j], d);
			}
		}
		for (int j = length/2; j > 0; j >>= 1) {
			__syncthreads();
			int pos = 2 * threadIdx.x - (threadIdx.x&(j - 1));
			swap(buf[pos], buf[pos + j], dir);
		}
		__syncthreads();
		res[threadIdx.x] = buf[threadIdx.x];
		res[threadIdx.x + MAX_LENGTH / 2] = buf[threadIdx.x + MAX_LENGTH / 2];
	}

	template<typename T>
	__global__ void firstBinoticSort(T *arr, T *res) {
		__shared__ T buf[MAX_LENGTH];
		int id = blockIdx.x * MAX_LENGTH + threadIdx.x;
		buf[threadIdx.x] = arr[id];
		buf[threadIdx.x + (MAX_LENGTH / 2)] = arr[id + (MAX_LENGTH / 2)];
		//printf("%d\n", arr[threadIdx.x]);
		//printf("%d\n", arr[threadIdx.x+ (MAX_LENGTH / 2)]);
		__syncthreads();
		for (int size = 2; size < MAX_LENGTH; size <<= 1) {
			int d = (threadIdx.x & (size / 2)) != 0;
			for (int j = size / 2; j > 0; j >>= 1) {
				__syncthreads();
				int pos = 2 * threadIdx.x - (threadIdx.x&(j - 1));
				swap(buf[pos], buf[pos + j], d);
			}
		}
	//printf("%d\n", blockIdx.x);
		int d = blockIdx.x & 1; //奇偶
		for (int j = MAX_LENGTH / 2; j > 0; j >>= 1) {
			__syncthreads();
			int pos = 2 * threadIdx.x - (threadIdx.x&(j - 1));
			swap(buf[pos], buf[pos + j], d);
		}
		__syncthreads();
		res[id] = buf[threadIdx.x];
		res[id + MAX_LENGTH / 2] = buf[threadIdx.x + MAX_LENGTH / 2];
		//printf("%d\n", res[threadIdx.x]);
	}

	template<typename T>
	__global__ void bitonicMergeLarge(T *arr, T *res, int length,int size,int j, int dir) {
		int id = blockIdx.x*blockDim.x + threadIdx.x;
		int com = id & (length/2 - 1);

		int d = dir^((com&(size / 2)) != 0);
		int pos = 2 * id - (id&(j - 1));

		int num1 = arr[pos];
		int num2 = arr[pos + j];
		swap(num1, num2, d);

		res[pos] = num1;
		res[pos + j] = num2;
	}

	template<typename T>
	__global__ void bitonicMergeSmall(T *arr, T *res, int length, int size, int dir) {
		__shared__ T buf[MAX_LENGTH];
		int id = blockIdx.x*MAX_LENGTH + threadIdx.x;
		buf[threadIdx.x] = arr[id];
		buf[threadIdx.x + MAX_LENGTH / 2] = arr[id + MAX_LENGTH / 2];

		int id1 = blockIdx.x * blockDim.x + threadIdx.x;
		int com = id1 & ((length / 2) - 1);
		int d = dir ^ ((com&(size / 2)) != 0);

		for (int j = MAX_LENGTH / 2; j > 0; j >>= 1) {
			__syncthreads();
			int pos = 2 * threadIdx.x - (threadIdx.x&(j - 1));
			swap(buf[pos], buf[pos + j], d);

		}
		__syncthreads();

		res[id] = buf[threadIdx.x];
		res[id + MAX_LENGTH / 2] = buf[threadIdx.x + MAX_LENGTH / 2];
	}

	/*
	arr：待排序数组
	length：数组长度
	dir：控制递增或者递减，1为递增，0为递减
	*/

	template<typename T>
	void SortCuda<T>::binoticSort(T * arr, int length, int dir){

		if (length <= MAX_LENGTH) {
			dataMalloc(MAX_LENGTH);

			arrExpand(arr, host_input, length, MAX_LENGTH);

			cudaMemcpy(device_input, host_input, MAX_LENGTH * sizeof(T), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			smallBinoticSort<T> << <1, MAX_LENGTH/2 >> > (device_input, device_res, MAX_LENGTH, dir);
			cudaDeviceSynchronize();
			cudaMemcpy(host_res, device_res, MAX_LENGTH * sizeof(T), cudaMemcpyDeviceToHost);

			if (dir == 1) {
				for (int i = 0; i < length; i++) {
					arr[i] = host_res[i];
				}
			}else if (dir == 0) {
				for (int i = MAX_LENGTH-length , j=0; i <MAX_LENGTH; i++,j++) {
					arr[j] = host_res[i];
				}
			}
			/*for (uint i = 0; i < length; i++) {
				printf("%d\n", arr[i]);
			}
			printf("\n");*/

		}
		else {
			if (length&(length-1)==0) {
				dataMalloc(length);
				for (int i = 0; i < length; i++) {
					host_input[i] = arr[i];
				}
				cudaMemcpy(device_input, host_input, length * sizeof(T), cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				int blockNum = length / MAX_LENGTH;
				int threadNum = MAX_LENGTH / 2;
				//printf("%d\n\n", length);
				firstBinoticSort<T> << <blockNum, threadNum >> > (device_input, device_res);

				for (int size = 2 * MAX_LENGTH; size <= length; size <<= 1) {
					for (int j = size / 2; j > 0; j >>= 1) {
						if (j >= MAX_LENGTH) {
							bitonicMergeLarge<T> << <length / MAX_LENGTH, MAX_LENGTH/2 >> > (device_res, device_res, length, size, j, dir);
						}
						else {
							bitonicMergeSmall<T> << <blockNum, threadNum >> > (device_res, device_res, length, size, dir);
							break;
						}
					}
				}
				cudaDeviceSynchronize();
				cudaMemcpy(host_res, device_res, length * sizeof(T), cudaMemcpyDeviceToHost);

				for (int i = 0; i < length; i++) {
					arr[i] = host_res[i];
				}
			}else{
				int temp = log(length) / log(2);
				int new_len = pow(2,temp+1);
				//printf("%d\n", new_len);
				dataMalloc(new_len);
				arrExpand(arr, this->host_input, length, new_len);
				cudaMemcpy(device_input, host_input, new_len * sizeof(T), cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				int blockNum = new_len / MAX_LENGTH;
				int threadNum = MAX_LENGTH / 2;

				firstBinoticSort<T> << <blockNum, threadNum >> > (device_input, device_res);

				for (int size = 2 * MAX_LENGTH; size <= new_len; size <<= 1) {
					for (int j = size / 2; j > 0; j >>= 1) {
						if (j >= MAX_LENGTH) {
							bitonicMergeLarge<T> << <new_len / MAX_LENGTH, MAX_LENGTH/2 >> > (device_res, device_res, new_len, size, j, dir);
						}
						else {
							bitonicMergeSmall<T> << <blockNum, threadNum >> > (device_res, device_res, new_len, size, dir);
							break;
						}
					}
				}
				cudaDeviceSynchronize();
				cudaMemcpy(host_res, device_res, new_len * sizeof(T), cudaMemcpyDeviceToHost);
				if (dir == 1) {
					for (int i = 0; i < length; i++) {
						arr[i] = host_res[i];
					}
				}
				else if (dir == 0) {
					for (int i = new_len - length, j = 0; i < new_len; i++, j++) {
						arr[j] = host_res[i];
					}
				}
			}

		}


	}

	template<typename T>
	void SortCuda<T>::dataMalloc(T length)
	{
		if (this->device_input == nullptr) {
			cudaFree(this->device_input);
		}
		if (this->device_res == nullptr) {
			cudaFree(this->device_res);
		}
		if (this->host_input == nullptr) {
			free(this->host_input);
		}
		if (this->host_res == nullptr) {
			free(this->host_res);
		}
		this->host_res = (T*)malloc(sizeof(T)*length);
		this->host_input = (T*)malloc(sizeof(T)*length);
		cudaMalloc((void **)&this->device_input, length * sizeof(T));
		cudaMalloc((void **)&this->device_res, length * sizeof(T));
	}

	template<typename T>
	void SortCuda<T>::arrExpand(T * arr, T * arr_new, int length, int new_length)
	{
		int com1 = 1;
		float com2 = 0;
		double com3 = 0.5;
		for (int i = 0; i < length; i++) {
			arr_new[i] = arr[i];
		}
		//const type_info &nInfo = typeid(T);
		//char* name = ;
		//printf("%s\n", nInfo.name());
		if (typeid(T)==typeid(com1) ){
			printf("-----------------------------------\n");
			for (int i = length; i < new_length; i++) {
				arr_new[i] = INT_MAX;
			}
		}
		else if (typeid(T) == typeid(com2)) {
			for (int i = length; i < new_length; i++) {
				arr_new[i] = FLT_MAX;
			}
		}
		else if (typeid(T) == typeid(com3)) {
			for (int i = length; i < new_length; i++) {
				arr_new[i] = DBL_MAX;
			}
		}
		/*switch (name) {
			case "int":		
				for (int i = length; i < new_length; i++) {
					arr_new[i] = INT_MAX;
				}
			case "float":
				for (int i = length; i < new_length; i++) {
					arr_new[i] = FLT_MAX;
				}
			case "double":
				for (int i = length; i < new_length; i++) {
					arr_new[i] = DBL_MAX;
				}
		}*/

	}

	
	/*
	int SortCuda::int2bit(int n) {
		int count = 0;
		while (n != 0) {
			n = n / 2;
			count++;
		}
		return count;
	}



	void SortCuda::numMalloc(int *arr, int length) {
		cudaMalloc((void**)&num, length * sizeof(int));
		cudaMemcpy((void*)num, (void*)arr, length * sizeof(int), cudaMemcpyHostToDevice);
	}*/
}