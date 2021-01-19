#include "gtest/gtest.h"
#include "Core/Utility/Sort.h"

using namespace PhysIKA;

TEST(Sort, function)
{
	int arr[9] = {4,9,1,8,3,2,7,6,5};
	int target[9] = { 1,2,3,4,5,6,7,8,9 };
	Sort s(9);
	s.radixSort(arr, 9);
	EXPECT_EQ(arr, target);
}
