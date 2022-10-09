#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>

#include "Trace.h"

namespace Bootstrap
{
	Eigen::MatrixXcd Expand(const Eigen::MatrixXcd& mat, Eigen::Index row, Eigen::Index col)
	{
		assert(mat.rows() <= row && mat.cols() <= col);

		Eigen::MatrixXcd res(row, col);
		res.setZero();
		res.block(0, 0, mat.rows(), mat.cols()) = mat;
		return res;
	}

	Eigen::MatrixXcd ExpectReal(const Eigen::MatrixXcd& data)
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
			result.setZero();
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
			result.setZero();
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

		bool Contains(const std::basic_string<Matrix>& op)
		{
			return this->index.contains(op);
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

		void AddConstraints(const Eigen::MatrixXcd& constraints)
		{
			assert(constraints.cols() == NumVariables());
			this->constraints = VStack(this->constraints, ExpectReal(constraints));
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

		Eigen::RowVectorXcd GetSolution(const std::basic_string<Matrix>& key)
		{
			return Eigen::RowVectorXcd(this->matrix.row(this->index.at(key)));
		}

		void AddSolution(const std::basic_string<Matrix>& key, const Eigen::RowVectorXcd& data)
		{
			assert(data.cols() == NumVariables());
			assert(!(this->index.contains(key)));

			this->index.insert(std::pair(key, NumOperators()));
			this->matrix = VStack(this->matrix, ExpectReal(data));
		}
	};

	struct OperatorPair
	{
		std::basic_string<Matrix> Op1;
		std::basic_string<Matrix> Op2;

		OperatorPair()
			: Op1(""), Op2("")
		{

		}

		OperatorPair(const std::basic_string<Matrix>& op1, const std::basic_string<Matrix>& op2)
			: Op1(op1), Op2(op2)
		{

		}

		class hash
		{
		public:
			size_t operator()(const Bootstrap::OperatorPair& p) const
			{
				using std::hash;
				return hash<std::string>()(p.Op1) * hash<std::string>()(p.Op2);
			}
		};
	};

	struct QuadraticOperator
	{
		complex Coefficient;
		OperatorPair Ops;

		QuadraticOperator()
			: Coefficient(), Ops()
		{

		}

		QuadraticOperator(const complex& coeff, const OperatorPair& ops)
			: Coefficient(coeff), Ops(ops)
		{

		}
	};

	bool operator==(const OperatorPair& p1, const OperatorPair& p2)
	{
		return (p1.Op1 == p2.Op1 && p1.Op2 == p2.Op2) ||
			(p1.Op1 == p2.Op2 && p1.Op2 == p2.Op1);
	}

	class QuadraticSolution
	{
	private:
		LinearSolution sol;
		std::unordered_map<OperatorPair, Eigen::Index, OperatorPair::hash> index;
		Eigen::MatrixXcd param1;
		Eigen::MatrixXcd param2;
		Eigen::MatrixXcd matrixLine;
		Eigen::MatrixXcd matrixQuad;

	public:
		QuadraticSolution(LinearSolution& sol)
			: sol(sol), index(), param1(0, sol.NumVariables()),
			param2(0, sol.NumVariables()), matrixLine(0, sol.NumVariables()),
			matrixQuad(0, 0)
		{

		}

		QuadraticSolution(LinearSolution&& sol)
			: QuadraticSolution(static_cast<LinearSolution&>(sol))
		{

		}

		Eigen::Index NumConstraints()
		{
			assert(this->matrixQuad.rows() == this->matrixLine.rows());
			return this->matrixQuad.rows();
		}

		Eigen::Index GetVariable(const std::basic_string<Matrix>& op1, const std::basic_string<Matrix>& op2)
		{
			auto pair = OperatorPair(op1, op2);
			if(this->index.contains(pair))
			{
				return this->index.at(pair);
			}

			Eigen::Index ind = this->index.size();
			this->param1 = VStack(this->param1, this->sol.GetSolution(op1));
			this->param2 = VStack(this->param2, this->sol.GetSolution(op2));
			this->matrixQuad = Expand(this->matrixQuad, this->matrixQuad.rows(), ind + 1);

			return ind;
		}

		void AddConstraint(const std::vector<QuadraticOperator>& quad, const Eigen::RowVectorXcd& linear)
		{
			this->matrixLine = VStack(this->matrixLine, ExpectReal(linear));

			Eigen::RowVectorXcd res(1, this->matrixQuad.cols());
			res.setZero();
			for(const auto& e : quad)
			{
				res(0, GetVariable(e.Ops.Op1, e.Ops.Op2)) = e.Coefficient;
			}

			this->matrixQuad = VStack(this->matrixQuad, ExpectReal(res));
		}

		void ReduceConstraints()
		{
			assert(this->matrixQuad.cols() == this->index.size());
			assert(this->matrixLine.cols() == this->sol.NumVariables());

			if(this->matrixQuad.rows() == 0)
				return;

			Eigen::MatrixXcd mat(HStack(this->matrixQuad, this->matrixLine));
			mat = RowSpace(mat);
			this->matrixQuad = mat.block(0, 0, mat.rows(), this->index.size());
			this->matrixLine = mat.block(0, this->index.size(), mat.rows(), mat.cols() - this->index.size());

			assert(this->matrixLine.cols() == this->sol.NumVariables());
		}
	};

	using ConjugateFunc = std::basic_string<Matrix>(*)(const std::basic_string<Matrix>&);

	template<uint32_t dimension>
	class Solver
	{
	private:
		TraceOperator<dimension> hamil;
		std::vector<Trace> gauge;
		Matrix mats[dimension];
		ConjugateFunc conj;
		LinearSolution solution;

	public:
		Solver(
			const TraceOperator<dimension>& hamil,
			const std::vector<Trace>& gauge,
			Matrix mats[dimension],
			ConjugateFunc conj
		)
			: hamil(hamil), gauge(gauge), conj(conj), solution()
		{
			for(size_t i = 0; i < dimension; i++)
			{
				this->mats[i] = mats[i];
			}
		}

	private:
		bool SolveHamiltonianConstraint(
			const std::basic_string<Matrix>& op,
			Eigen::RowVectorXcd* result
		)
		{
			std::vector<Trace> ops(1);
			ops[0] = Trace(complex(1.0, 0.0), op);
			TraceOperator<dimension> commutator = this->hamil.Commutator(
				TraceOperator<dimension>(
				this->hamil.matInfo, ops
			)).Rewrite(this->mats);

			Eigen::RowVectorXcd res(1, this->solution.NumVariables());
			double coef = 0.0;

			for(auto& tr : commutator.ops)
			{
				auto& c = tr.Coefficient;
				auto& s = tr.Matrices;
				if(tr.Matrices == op)
					coef += c.real();
				else
				{
					assert(s.size() > op.size());
					if(!(this->solution.Contains(s)))
					{
						return false;
					}

					res += c.real() * this->solution.GetSolution(s);
				}
			}

			if(std::abs(coef) < 1e-8)
			{
				this->solution.AddConstraints(res);
				return false;
			}

			*result = (-1.0 / coef) * res;
			return true;
		}

		bool ImposeGaugeConstraint(const std::basic_string<Matrix>& op)
		{
			Eigen::RowVectorXcd res(1, this->solution.NumVariables());
			
			for(auto& tr : this->gauge)
			{
				auto& c = tr.Coefficient;
				auto& s = tr.Matrices;

				TraceOperator<dimension> opNew = TraceOperator<dimension>(
					this->hamil.matInfo, s + op
					).Rewrite(this->mats);

				for(auto& ntr : opNew.ops)
				{
					auto& d = ntr.Coefficient;
					auto& t = ntr.Matrices;

					if(!(this->solution.Contains(t)))
						return false;
					
					res += c * d * this->solution.GetSolution(t);
				}
			}

			this->solution.AddConstraints(res);
			return true;
		}
	};
}