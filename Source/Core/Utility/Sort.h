#pragma once
namespace PhysIKA {

	class Sort{
		public:
			Sort();
			~Sort();
			Sort(int length);
			void radixSort(int *arr, int length);

		private:

			void numMalloc(int *arr,int length);
			int sortLength;
			int * num;
			int int2bit(int n);
	};


}