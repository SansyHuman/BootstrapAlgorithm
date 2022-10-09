#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>

#include "Trace.h"

namespace Bootstrap
{
	Eigen::MatrixXcd Expand(Eigen::MatrixXcd& mat, Eigen::Index row, Eigen::Index col)
	{
		assert(mat.rows() <= row && mat.cols() <= col);

		Eigen::MatrixXcd res(row, col);
		res.setZero();
		res.block(0, 0, mat.rows(), mat.cols()) = mat;
		return res;
	}

	Eigen::MatrixXcd Expand(Eigen::MatrixXcd&& mat, Eigen::Index row, Eigen::Index col)
	{
		return Expand(static_cast<Eigen::MatrixXcd&>(mat), row, col);
	}

	Eigen::MatrixXcd ExpectReal(Eigen::MatrixXcd& data)
	{
		int cntImagNonzero = 0;
		int cntRealNonzero = 0;

		for(Eigen::Index i = 0; i < data.rows(); i++)
		{
			for(Eigen::Index j = 0; j < data.cols(); j++)
			{
				auto& e = data(i, j);
				if(e.imag() != 0.0)
					cntImagNonzero++;
				if(e.real() != 0.0)
					cntRealNonzero++;
			}
		}

		if(cntImagNonzero == 0)
		{
			Eigen::MatrixXcd result(data.rows(), data.cols());
			for(Eigen::Index i = 0; i < data.rows(); i++)
			{
				for(Eigen::Index j = 0; j < data.cols(); j++)
				{
					result(i, j) = complex(data(i, j).real(), 0.0);
				}
			}

			return result;
		}

		if(cntRealNonzero == 0)
		{
			Eigen::MatrixXcd result(data.rows(), data.cols());
			for(Eigen::Index i = 0; i < data.rows(); i++)
			{
				for(Eigen::Index j = 0; j < data.cols(); j++)
				{
					result(i, j) = complex(data(i, j).imag(), 0.0);
				}
			}

			return result;
		}

		std::cerr << "WARNING: there are complex entries in the data." << std::endl;

		return Eigen::MatrixXcd(data);
	}

	Eigen::MatrixXcd ExpectReal(Eigen::MatrixXcd&& data)
	{
		return ExpectReal(static_cast<Eigen::MatrixXcd&>(data));
	}

	class LinearSolution
	{
	private:
		std::unordered_map<std::basic_string<Matrix>, Eigen::Index> index;
		Eigen::MatrixXcd matrix;
		Eigen::MatrixXcd constraints;

	public:
		LinearSolution()
			: index(), matrix(0, 0), constraints(0, 0)
		{

		}

		Eigen::Index NumVariables()
		{
			assert(this->matrix.cols() == this->constraints.cols());
			return this->matrix.cols();
		}

		Eigen::Index NumOperators()
		{
			assert(this->matrix.rows() == this->index.size());
			return this->matrix.rows();
		}

		Eigen::RowVectorXcd GetVariable()
		{
			auto num = NumVariables();
			Eigen::RowVectorXcd res(1, num + 1);
			res.setZero();
			res(0, num) = complex(1.0, 0.0);

			this->matrix = Expand(this->matrix, this->matrix.rows(), num + 1);
			this->constraints = Expand(this->constraints, this->constraints.rows(), num + 1);

			return res;
		}

		void AddConstraints(Eigen::MatrixXcd& constraints)
		{
			assert(constraints.cols() == NumVariables());
			this->constraints = VStack(this->constraints, ExpectReal(constraints));
		}

		void AddConstraints(Eigen::MatrixXcd&& constraints)
		{
			AddConstraints(static_cast<Eigen::MatrixXcd&>(constraints));
		}

		void SolveConstraints()
		{
			if(this->constraints.rows() == 0 || NumVariables() == 0)
				return;

			auto sol = NullSpace(this->constraints);
			this->matrix = this->matrix * sol;
			this->constraints.resize(0, sol.cols());
			this->constraints.setZero();
		}

		Eigen::RowVectorXcd GetSolution(std::basic_string<Matrix>& key)
		{
			return Eigen::RowVectorXcd(this->matrix.row(this->index.at(key)));
		}

		Eigen::RowVectorXcd GetSolution(std::basic_string<Matrix>&& key)
		{
			return GetSolution(static_cast<std::basic_string<Matrix>&>(key));
		}

		void AddSolution(std::basic_string<Matrix>& key, Eigen::RowVectorXcd& data)
		{
			assert(data.cols() == NumVariables());
			assert(!(this->index.contains(key)));

			this->index.insert(std::pair(key, NumOperators()));
			this->matrix = VStack(this->matrix, ExpectReal(data));
		}

		void AddSolution(std::basic_string<Matrix>&& key, Eigen::RowVectorXcd& data)
		{
			AddSolution(static_cast<std::basic_string<Matrix>&>(key), data);
		}

		void AddSolution(std::basic_string<Matrix>& key, Eigen::RowVectorXcd&& data)
		{
			AddSolution(key, static_cast<Eigen::RowVectorXcd&>(data));
		}

		void AddSolution(std::basic_string<Matrix>&& key, Eigen::RowVectorXcd&& data)
		{
			AddSolution(static_cast<std::basic_string<Matrix>&>(key), static_cast<Eigen::RowVectorXcd&>(data));
		}
	};
}