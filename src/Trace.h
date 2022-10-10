#pragma once

#include <vector>
#include <array>
#include <cassert>
#include <unordered_set>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/LU>
#include <string>
#include <initializer_list>
#include <unordered_map>

#include "Utils.h"

namespace Bootstrap
{
	using Matrix = char;
	using complex = std::complex<double>;

	template<uint32_t dimension> class TraceOperator;

	template<uint32_t dimension>
	class MatrixInfo
	{
	private:
		std::vector<std::array<Matrix, dimension>> matrices;
		std::vector<Eigen::Matrix<complex, dimension, dimension>> coefficients;
		Eigen::Matrix<complex, dimension, dimension> commutators;

		friend class TraceOperator<dimension>;

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
				size_t index = Find(ms, matrix, &contains);
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

		size_t BasisIndex(Matrix matrices[dimension], bool* contains)
		{
			for(size_t i = 0; i < this->matrices.size(); i++)
			{
				std::array<Matrix, dimension>& basis = this->matrices[i];
				bool match = true;
				for(size_t j = 0; j < dimension; j++)
				{
					if(matrices[j] != basis[j])
					{
						match = false;
						break;
					}
				}
				if(match)
				{
					*contains = true;
					return i;
				}
			}

			*contains = false;
			return -1;
		}

		bool Equals(const MatrixInfo<dimension>& other)
		{
			if(this->matrices.size() != other.matrices.size())
				return false;

			for(size_t i = 0; i < this->matrices.size(); i++)
			{
				if(!ArrayEquals(this->matrices[i], other.matrices[i]))
					return false;
			}

			return true;
		}
	};

	struct Trace
	{
		complex Coefficient;
		std::basic_string<Matrix> Matrices;

		Trace()
			: Coefficient({ 0.0, 0.0 }), Matrices("")
		{}

		Trace(const complex& coeff, const std::basic_string<Matrix>& mats)
			: Coefficient(coeff), Matrices(mats)
		{

		}
	};

	std::ostream& operator<<(std::ostream& os, const Trace& tr)
	{
		return os << "(" << tr.Coefficient.real() << " + " << tr.Coefficient.imag() << "i" << ")tr" << tr.Matrices;
	}

	template<uint32_t dimension>
	class Solver;

	template<uint32_t dimension>
	class TraceOperator
	{
	private:
		MatrixInfo<dimension>* matInfo;
		std::vector<Trace> ops;

		friend std::ostream& operator<<<dimension>(std::ostream&, const TraceOperator<dimension>&);
		friend class Bootstrap::Solver<dimension>;

	public:
		TraceOperator(const MatrixInfo<dimension>& matInfo, const std::vector<Trace>& ops)
			: matInfo(const_cast<MatrixInfo<dimension>*>(&matInfo)), ops(0)
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

		TraceOperator<dimension> operator-()
		{
			std::vector<Trace> newOps(this->ops);
			for(auto& op : newOps)
			{
				op.Coefficient = -op.Coefficient;
			}

			return TraceOperator<dimension>(*(this->matInfo), newOps);
		}

		TraceOperator<dimension> Rewrite(Matrix matrices[dimension])
		{
			bool contains = false;
			size_t index = this->matInfo->BasisIndex(matrices, &contains);
			assert(contains);

			Eigen::Matrix<complex, dimension, dimension> table(this->matInfo->coefficients[index].inverse());

			std::vector<Trace> inProcess = this->ops;
			std::unordered_map<std::basic_string<Matrix>, complex> finalResult;
			while(inProcess.size() > 0)
			{
				std::vector<Trace> result(0);

				for(auto& tr : inProcess)
				{
					bool toBeConverted = false;
					size_t convertIndex = -1;

					for(size_t i = 0; i < tr.Matrices.size(); i++)
					{
						bool contains = false;
						size_t index = Find<Matrix, dimension>(&matrices[0], tr.Matrices.at(i), &contains);
						if(!contains)
						{
							toBeConverted = true;
							convertIndex = i;
							break;
						}
					}

					if(!toBeConverted)
					{
						if(finalResult.contains(tr.Matrices))
						{
							finalResult[tr.Matrices] += tr.Coefficient;
						}
						else
						{
							finalResult.insert(std::pair(tr.Matrices, tr.Coefficient));
						}
					}
					else
					{
						size_t pos = convertIndex;
						auto coef = Eigen::Matrix<complex, 1, dimension>(
							this->matInfo->GetCoefficients(tr.Matrices.at(convertIndex)).transpose()
							* table
							);

						for(size_t i = 0; i < dimension; i++)
						{
							Matrix m = matrices[i];
							result.push_back(Trace(
								tr.Coefficient * coef(0, i),
								tr.Matrices.substr(0, pos) + m + tr.Matrices.substr(pos + 1)
							));
						}
					}
				}

				inProcess = result;
			}

			std::vector<Trace> finalTr;
			for(auto& tr : finalResult)
			{
				finalTr.push_back(Trace(tr.second, tr.first));
			}

			return TraceOperator<dimension>(*(this->matInfo), finalTr);
		}

		TraceOperator<dimension> Commutator(const TraceOperator<dimension>& other)
		{
			assert(this->matInfo->Equals(*(other.matInfo)));

			std::vector<Trace> res(0);

			for(auto& tr1 : this->ops)
			{
				for(auto& tr2 : other.ops)
				{
					for(size_t i = 0; i < tr1.Matrices.size(); i++)
					{
						Matrix m1 = tr1.Matrices.at(i);
						for(size_t j = 0; j < tr2.Matrices.size(); j++)
						{
							Matrix m2 = tr2.Matrices.at(j);
							for(auto& o1 : tr1.Matrices.substr(0, i))
							{
								for(auto& o2 : tr1.Matrices.substr(i + 1))
								{
									assert(std::abs(this->matInfo->Commutator(o1, o2)) < 1e-8);
								}
							}

							std::basic_string<Matrix> s =
								tr2.Matrices.substr(0, j) +
								tr1.Matrices.substr(i + 1) +
								tr1.Matrices.substr(0, i) +
								tr2.Matrices.substr(j + 1);
							res.push_back(Trace(
								tr1.Coefficient * tr2.Coefficient * this->matInfo->Commutator(m1, m2),
								s
							));
						}
					}
				}
			}

			return TraceOperator<dimension>(*(this->matInfo), res);
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
}