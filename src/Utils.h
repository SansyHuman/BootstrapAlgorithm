#pragma once

#include <array>
#include <Eigen/Dense>

namespace Bootstrap
{
	template<class T, size_t size>
	inline size_t Find(const std::array<T, size>& arr, const T& element, bool* contains)
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
	inline size_t Find(const T* arr, const T& element, bool* contains)
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
	inline bool ArrayEquals(const std::array<T, size>& a1, const std::array<T, size>& a2)
	{
		for(size_t i = 0; i < size; i++)
		{
			if(a1[i] != a2[i])
				return false;
		}

		return true;
	}

	Eigen::MatrixXcd VStack(const Eigen::MatrixXcd& upper, const Eigen::MatrixXcd& lower)
	{
		assert(upper.cols() == lower.cols());

		Eigen::MatrixXcd res(upper.rows() + lower.rows(), upper.cols());
		res.setZero();
		res.block(0, 0, upper.rows(), upper.cols()) = upper;
		res.block(upper.rows(), 0, lower.rows(), lower.cols()) = lower;

		return res;
	}

	Eigen::MatrixXcd HStack(const Eigen::MatrixXcd& left, const Eigen::MatrixXcd& right)
	{
		assert(left.rows() == right.rows());

		Eigen::MatrixXcd res(left.rows(), left.cols() + right.cols());
		res.setZero();
		res.block(0, 0, left.rows(), left.cols()) = left;
		res.block(0, left.cols(), right.rows(), right.cols()) = right;

		return res;
	}
}