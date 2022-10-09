#pragma once

#include <array>
#include <Eigen/Dense>

namespace Bootstrap
{
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
	inline size_t Find(const T* arr, T& element, bool* contains)
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

	Eigen::MatrixXcd VStack(Eigen::MatrixXcd& upper, Eigen::MatrixXcd& lower)
	{
		assert(upper.cols() == lower.cols());

		Eigen::MatrixXcd res(upper.rows() + lower.rows(), upper.cols());
		res.block(0, 0, upper.rows(), upper.cols()) = upper;
		res.block(upper.rows(), 0, lower.rows(), lower.cols()) = lower;

		return res;
	}

	Eigen::MatrixXcd VStack(Eigen::MatrixXcd& upper, Eigen::MatrixXcd&& lower)
	{
		return VStack(upper, static_cast<Eigen::MatrixXcd&>(lower));
	}

	Eigen::MatrixXcd VStack(Eigen::MatrixXcd&& upper, Eigen::MatrixXcd& lower)
	{
		return VStack(static_cast<Eigen::MatrixXcd&>(upper), lower);
	}

	Eigen::MatrixXcd VStack(Eigen::MatrixXcd&& upper, Eigen::MatrixXcd&& lower)
	{
		return VStack(static_cast<Eigen::MatrixXcd&>(upper), static_cast<Eigen::MatrixXcd&>(lower));
	}

	Eigen::MatrixXcd HStack(Eigen::MatrixXcd& left, Eigen::MatrixXcd& right)
	{
		assert(left.rows() == right.rows());

		Eigen::MatrixXcd res(left.rows(), left.cols() + right.cols());
		res.block(0, 0, left.rows(), left.cols()) = left;
		res.block(0, left.cols(), right.rows(), right.cols()) = right;

		return res;
	}

	Eigen::MatrixXcd HStack(Eigen::MatrixXcd& left, Eigen::MatrixXcd&& right)
	{
		return HStack(left, static_cast<Eigen::MatrixXcd&>(right));
	}

	Eigen::MatrixXcd HStack(Eigen::MatrixXcd&& left, Eigen::MatrixXcd& right)
	{
		return HStack(static_cast<Eigen::MatrixXcd&>(left), right);
	}

	Eigen::MatrixXcd HStack(Eigen::MatrixXcd&& left, Eigen::MatrixXcd&& right)
	{
		return HStack(static_cast<Eigen::MatrixXcd&>(left), static_cast<Eigen::MatrixXcd&>(right));
	}
}