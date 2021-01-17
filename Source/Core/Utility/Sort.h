#pragma once
namespace PhysIKA {

	class Sort{
		public:
			Sort();
			~Sort();
			int * radixSort(int *arr, int length);

		private:
			Sort(int length);
			void numMalloc(int *arr,int length);
			int sortLength;
			int * num;
			int int2bit(int n);
	};


}