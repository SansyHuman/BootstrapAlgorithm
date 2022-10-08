#pragma once

#include <vector>
#include <array>
#include <cassert>
#include <unordered_set>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/QR>

#include "Utils.h"

using Matrix = char;
using complex = std::complex<double>;

template<uint32_t dimension>
class MatrixInfo
{
private:
	std::vector<std::array<Matrix, dimension>> matrices;
	std::vector<Eigen::Matrix<complex, dimension, dimension>> coefficients;
	Eigen::Matrix<complex, dimension, dimension> commutators;

private:
	static void MatrixDuplicateCheck(Matrix matrices[dimension])
	{
		std::unordered_set<Matrix> tmp;
		for(uint32_t i = 0; i < dimension; i++)
		{
			tmp.insert(matrices[i]);
		}
		assert(tmp.size() == dimension);
	}

public:
	MatrixInfo(Matrix matrices[dimension], Eigen::Matrix<complex, dimension, dimension>& commutators)
	{
		MatrixDuplicateCheck(matrices);

		this->matrices.resize(1);
		for(uint32_t i = 0; i < dimension; i++)
		{
			this->matrices[0][i] = matrices[i];
		}

		coefficients.clear();
		Eigen::Matrix<complex, dimension, dimension> eye;
		eye.setIdentity();
		coefficients.push_back(eye);

		this->commutators = commutators;
	}

	void AddBasis(Matrix matrices[dimension], Eigen::Matrix<complex, dimension, dimension>& coefficients)
	{
		assert(coefficients.colPivHouseholderQr().rank() == dimension);
		MatrixDuplicateCheck(matrices);

		this->matrices.push_back(std::array<Matrix, dimension>());
		size_t last = this->matrices.size() - 1;
		for(uint32_t i = 0; i < dimension; i++)
		{
			this->matrices[last][i] = matrices[i];
		}

		this->coefficients.push_back(coefficients);
	}

	Eigen::Matrix<complex, dimension, 1> GetCoefficients(Matrix matrix)
	{
		for(size_t i = 0; i < matrices.size(); i++)
		{
			auto& ms = matrices[i];
			bool contains = false;
			size_t index = Contains(ms, matrix, &contains);
			if(contains)
			{
				return Eigen::Matrix<complex, dimension, 1>(coefficients[i].row(index).transpose());
			}
		}

		assert(false);
	}

	complex Commutator(Matrix m1, Matrix m2)
	{
		auto c1 = GetCoefficients(m1);
		auto c2 = GetCoefficients(m2);
		auto t1 = Eigen::Matrix<complex, dimension, 1>(commutators * c2);
		complex result = (c1.transpose() * t1)(0, 0);
		return result;
	}
};