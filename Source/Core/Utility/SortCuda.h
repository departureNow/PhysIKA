#pragma once

namespace PhysIKA {
	//typedef unsigned int uint;

	template<typename T>
	class SortCuda{
		public:
			SortCuda();
			SortCuda(int length);
			~SortCuda();
			//void radixSort(int *arr, int length);
			void binoticSort(T *arr, int length, int dir);
		private:

			//void numMalloc(int *arr,int length);
			int sortLength;
			T * host_res;
			T * host_input;
			T * device_input;
			T * device_res;
			void dataMalloc(T length);
			void arrExpand(T * arr, T * arr_new, int length, int new_length);
			//int int2bit(int n);
	};

	template class SortCuda<int>;
	template class SortCuda<float>;
	template class SortCuda<double>;


}