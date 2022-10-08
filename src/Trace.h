#pragma once

#include <vector>
#include <array>
#include <cassert>
#include <unordered_set>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <string>
#include <initializer_list>
#include <unordered_map>

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

struct Trace
{
	complex Coefficient;
	std::basic_string<Matrix> Matrices;

	Trace()
		: Coefficient({0.0, 0.0}), Matrices("")
	{}

	Trace(complex coeff, std::basic_string<Matrix> mats)
		: Coefficient(coeff), Matrices(mats)
	{

	}
};

std::ostream& operator<<(std::ostream& os, const Trace& tr)
{
	return os << "(" << tr.Coefficient.real() << " + " << tr.Coefficient.imag() << "i" << ")tr" << tr.Matrices;
}

template<uint32_t dimension>
class TraceOperator
{
private:
	MatrixInfo<dimension> matInfo;
	std::vector<Trace> ops;

	friend std::ostream& operator<<<dimension>(std::ostream&, const TraceOperator<dimension>&);

public:
	TraceOperator(MatrixInfo<dimension>& matInfo, const std::vector<Trace>& ops)
		: matInfo(matInfo), ops(0)
	{
		auto coef = std::unordered_map<std::basic_string<Matrix>, complex>();
		for(auto& op : ops)
		{
			if(coef.contains(op.Matrices))
			{
				coef[op.Matrices] += op.Coefficient;
			}
			else
			{
				coef.insert(std::pair(op.Matrices, op.Coefficient));
			}
		}

		for(auto& s : coef)
		{
			if(std::abs(s.second) > 1e-8)
				this->ops.push_back(Trace(s.second, s.first));
		}
	}

	TraceOperator(MatrixInfo<dimension>& matInfo, std::vector<Trace>&& ops)
		: TraceOperator(matInfo, ops)
	{

	}

	TraceOperator(MatrixInfo<dimension>& matInfo, const std::initializer_list<Trace>& ops)
		: TraceOperator(matInfo, std::vector<Trace>{ops})
	{
	}

	TraceOperator<dimension> operator-()
	{
		std::vector<Trace> newOps(this->ops);
		for(auto& op : newOps)
		{
			op.Coefficient = -op.Coefficient;
		}

		return TraceOperator<dimension>(this->matInfo, newOps);
	}

	TraceOperator<dimension> Rewrite(Matrix matrices[dimension])
	{

	}
};

template<uint32_t dimension>
std::ostream& operator<<(std::ostream& o, const TraceOperator<dimension>& tr)
{
	for(auto& op : tr.ops)
	{
		o << " + " << op;
	}

	return o;
}