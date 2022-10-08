#pragma once

#include <array>

template<class T, size_t size>
inline size_t Contains(std::array<T, size>& arr, T& element, bool* contains)
{
	for(size_t i = 0; i < size; i++) {
		if(arr[i] == element)
		{
			*contains = true;
			return i;
		}
	}

	*contains = false;
	return -1;
}