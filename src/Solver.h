#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <set>

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
		auto imnz = (data.imag().array() != 0).count();
		auto renz = (data.real().array() != 0).count();

		if(imnz == 0)
			return data.real();
		if(renz == 0)
			return data.imag();

		std::cerr << "WARNING: there are complex entries in the data." << std::endl;

		return data;
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

		Eigen::Index NumVariables() const
		{
			assert(this->matrix.cols() == this->constraints.cols());
			return this->matrix.cols();
		}

		Eigen::Index NumOperators() const
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
		LinearSolution* sol;
		std::unordered_map<OperatorPair, Eigen::Index, OperatorPair::hash> index;
		Eigen::MatrixXcd param1;
		Eigen::MatrixXcd param2;
		Eigen::MatrixXcd matrixLine;
		Eigen::MatrixXcd matrixQuad;

	public:
		QuadraticSolution()
			: sol(nullptr), index(), param1(0, 0), param2(0, 0), matrixLine(0, 0),
			matrixQuad(0, 0)
		{

		}

		QuadraticSolution(const LinearSolution& sol)
			: sol(const_cast<LinearSolution*>(&sol)), index(),
			param1(0, sol.NumVariables()),
			param2(0, sol.NumVariables()), matrixLine(0, sol.NumVariables()),
			matrixQuad(0, 0)
		{

		}


		LinearSolution& Solution()
		{
			return *sol;
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
			this->index.insert(std::pair(OperatorPair(op1, op2), ind));
			this->param1 = VStack(this->param1, this->sol->GetSolution(op1));
			this->param2 = VStack(this->param2, this->sol->GetSolution(op2));
			this->matrixQuad = Expand(this->matrixQuad, this->matrixQuad.rows(), ind + 1);

			return ind;
		}

		void AddConstraint(const std::vector<QuadraticOperator>& quad, const Eigen::RowVectorXcd& linear)
		{
			this->matrixLine = VStack(this->matrixLine, ExpectReal(linear));

			std::unordered_map<Eigen::Index, complex> data;
			for(const auto& e : quad)
			{
				Eigen::Index columnIndex = GetVariable(e.Ops.Op1, e.Ops.Op2);
				data.insert(std::pair(columnIndex, e.Coefficient));
			}

			Eigen::RowVectorXcd res(1, this->matrixQuad.cols());
			res.setZero();
			for(const auto& e : data)
			{
				res(0, e.first) = e.second;
			}

			this->matrixQuad = VStack(this->matrixQuad, ExpectReal(res));
		}

		void ReduceConstraints()
		{
			assert(this->matrixQuad.cols() == this->index.size());
			assert(this->matrixLine.cols() == this->sol->NumVariables());

			if(this->matrixQuad.rows() == 0)
				return;

			Eigen::MatrixXcd mat(HStack(this->matrixQuad, this->matrixLine));
			mat = RowSpace(mat);

			this->matrixQuad = mat.block(0, 0, mat.rows(), this->index.size());
			this->matrixLine = mat.block(0, this->index.size(), mat.rows(), mat.cols() - this->index.size());

			assert(this->matrixLine.cols() == this->sol->NumVariables());
		}
	};

	using ConjugateFunc = std::basic_string<Matrix>(*)(const std::basic_string<Matrix>&);
	using DegreeFunc = int(*)(const std::basic_string<Matrix>&);

	template<uint32_t dimension>
	class Solver
	{
	private:
		TraceOperator<dimension>* hamil;
		std::vector<Trace> gauge;
		Matrix mats[dimension];
		ConjugateFunc conj;
		LinearSolution solution;
		QuadraticSolution quad;

	public:
		Solver(
			const TraceOperator<dimension>& hamil,
			const std::vector<Trace>& gauge,
			Matrix mats[dimension],
			ConjugateFunc conj
		)
			: hamil(const_cast<TraceOperator<dimension>*>(&hamil)), gauge(gauge),
			conj(conj), solution(), quad()
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
			TraceOperator<dimension> commutator = this->hamil->Commutator(
				TraceOperator<dimension>(
				*(this->hamil->matInfo), ops
			)).Rewrite(this->mats);

			Eigen::RowVectorXcd res(1, this->solution.NumVariables());
			res.setZero();
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
			res.setZero();
			
			for(auto& tr : this->gauge)
			{
				auto& c = tr.Coefficient;
				auto& s = tr.Matrices;

				TraceOperator<dimension> opNew = TraceOperator<dimension>(
					*(this->hamil->matInfo), { Trace(complex(1.0), s + op) }
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

		bool CyclicityConstraint(
			const std::basic_string<Matrix>& op,
			std::vector<QuadraticOperator>* quad,
			Eigen::RowVectorXcd* linear,
			bool linearOnly = false
		)
		{
			if(op.size() < 2 ||
				!(this->solution.Contains(op)) ||
				!(this->solution.Contains(op.substr(1) + op.at(0))))
			{
				return false;
			}

			*linear = this->solution.GetSolution(op) - this->solution.GetSolution(op.substr(1) + op.at(0));
			if(quad != nullptr)
				quad->clear();

			for(size_t i = 0; i < op.size(); i++)
			{
				complex coef = this->hamil->matInfo->Commutator(op.at(0), op.at(i));
				assert(std::abs(coef.imag()) < 1e-8);

				double coefR = coef.real();

				if(i % 2 == 1 && std::abs(coefR) > 1e-8)
				{
					std::string op1 = op.substr(1, i - 1);
					std::string op2 = op.substr(i + 1);

					if(!(this->solution.Contains(op1)) || !(this->solution.Contains(op2)))
						return false;

					if(op1 == "" || op2 == "")
					{
						*linear -= coefR * this->solution.GetSolution(
							op1 != "" ? op1 : op2
						);
					}
					else if(linearOnly)
					{
						return false;
					}
					else
					{
						quad->push_back(QuadraticOperator(
							complex(-coefR, 0.0), OperatorPair(op1, op2)
						));
					}
				}
			}

			return true;
		}

		void Solve(const std::vector<std::basic_string<Matrix>>& ops)
		{
			std::unordered_set<std::basic_string<Matrix>> seen;
			for(auto& e : ops)
				seen.insert(e);

			std::vector<std::basic_string<Matrix>> operators;
			std::copy(seen.begin(), seen.end(), std::back_inserter(operators));
			std::sort(operators.begin(), operators.end(),
				[](std::basic_string<Matrix> o1, std::basic_string<Matrix> o2) {
					return o1.size() > o2.size();
				});

			for(size_t i = 0; i < operators.size(); i++)
			{
				auto& op = operators[i];

				assert(!(this->solution.Contains(op)));

				Eigen::RowVectorXcd res;
				res.setZero();
				if(this->solution.Contains(this->conj(op)))
				{
					res = this->solution.GetSolution(this->conj(op));
				}
				else
				{
					bool solvable = false;
					if(op.size() < operators[0].size())
					{
						solvable = this->SolveHamiltonianConstraint(op, &res);
					}

					if(!solvable)
					{
						res = this->solution.GetVariable();
					}
				}

				this->solution.AddSolution(op, res);
				this->ImposeGaugeConstraint(op);
			}

			for(size_t i = 0; i < operators.size(); i++)
			{
				auto& op = operators[i];

				Eigen::RowVectorXcd res;
				res.setZero();
				bool solvable = this->CyclicityConstraint(op, nullptr, &res, true);
				if(solvable)
				{
					this->solution.AddConstraints(res);
				}
			}

			this->solution.SolveConstraints();

			this->quad = QuadraticSolution(this->solution);

			for(size_t i = 0; i < operators.size(); i++)
			{
				auto& op = operators[i];

				std::vector<QuadraticOperator> quad;
				Eigen::RowVectorXcd lin;
				lin.setZero();
				bool solvable = this->CyclicityConstraint(op, &quad, &lin);
				if(solvable && quad.size() > 0)
				{
					this->quad.AddConstraint(quad, lin);
				}
			}

			this->quad.ReduceConstraints();
		}

		void GenerateString(std::vector<std::basic_string<Matrix>>* output, size_t maxlen)
		{
			output->clear();

			std::set<std::basic_string<Matrix>> prev;
			std::set<std::basic_string<Matrix>> current;

			output->push_back("");
			prev.insert("");

			for(size_t i = 1; i <= maxlen; i++)
			{
				current.clear();

				for(auto& s : prev)
				{
					for(size_t n = 0; n < dimension; n++)
					{
						current.insert(s + this->mats[n]);
					}
				}

				std::copy(current.begin(), current.end(), std::back_inserter(*output));
				prev.clear();
				std::copy(current.begin(), current.end(), std::inserter(prev, prev.begin()));
			}
		}

	public:
		void TableWithConstrinats(size_t maxlen, DegreeFunc degree,
			std::vector<std::pair<size_t, Eigen::MatrixXcd>>* tables,
			QuadraticSolution* constraint)
		{
			std::vector<std::basic_string<Matrix>> trialOps;
			GenerateString(&trialOps, maxlen);

			std::unordered_map<int, std::vector<std::basic_string<Matrix>>> trialOpsByDegree;
			for(auto& op : trialOps)
			{
				int d = degree(op);
				if(!trialOpsByDegree.contains(d))
				{
					trialOpsByDegree.insert(std::pair(d, std::vector<std::basic_string<Matrix>>(0)));
					
				}
				trialOpsByDegree.at(d).push_back(op);
			}

			std::vector<std::pair<size_t, std::vector<std::basic_string<Matrix>>>> tmpTables;
			for(auto& to : trialOpsByDegree)
			{
				int d = to.first;
				size_t len = trialOpsByDegree.at(d).size();
				std::vector<std::basic_string<Matrix>> opCombs;
				
				for(auto& s1 : trialOpsByDegree.at(d))
				{
					for(auto& s2 : trialOpsByDegree.at(d))
					{
						opCombs.push_back(this->conj(s1) + s2);
					}
				}

				tmpTables.push_back(std::pair(len, std::move(opCombs)));
			}

			std::vector<std::basic_string<Matrix>> ops;
			for(auto& e : tmpTables)
			{
				std::copy(e.second.begin(), e.second.end(), std::back_inserter(ops));
			}

			Solve(ops);

			tables->clear();

			for(auto& te : tmpTables)
			{
				size_t size = te.first;
				auto& table = te.second;

				std::vector<Eigen::MatrixXcd> solutions;
				for(auto& ent : table)
				{
					solutions.push_back(this->solution.GetSolution(ent));
				}

				tables->push_back(std::pair(size, VStack(solutions)));
			}

			*constraint = this->quad;
		}
	};
}