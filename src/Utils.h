#pragma once

#include <array>

template<class T, size_t size>
inline size_t Find(std::array<T, size>& arr, T& element, bool* contains)
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

template<class T, size_t size>
inline size_t Find(T* arr, T& element, bool* contains)
{
	for(size_t i = 0; i < size; i++)
	{
		if(arr[i] == element)
		{
			*contains = true;
			return i;
		}
	}

	*contains = false;
	return -1;
}

template<class T, size_t size>
inline bool ArrayEquals(std::array<T, size>& a1, std::array<T, size>& a2)
{
	for(size_t i = 0; i < size; i++)
	{
		if(a1[i] != a2[i])
			return false;
	}

	return true;
}