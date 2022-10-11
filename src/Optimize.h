#pragma once

#include <Eigen/Dense>
#include <Eigen/EigenValues>
#include <nlopt.hpp>
#include "Trace.h"
#include "Solver.h"

namespace Bootstrap
{
	template<uint32_t dimension>
	Eigen::RowVectorXcd OperatorToVector(LinearSolution& sol, TraceOperator<dimension>& ops)
	{
		Eigen::RowVectorXcd vec(1, sol.NumVariables());
		vec.setZero();

		for(const Trace& op : ops.Components())
		{
			vec += op.Coefficient.real() * sol.GetSolution(op.Matrices);
		}

		return vec;
	}

	double MinimalEigval(
		std::vector<std::pair<size_t, Eigen::MatrixXcd>>& tables,
		const Eigen::RowVectorXd& param
	)
	{
		std::vector<Eigen::MatrixXd> tableVals;
		for(auto& e : tables)
		{
			auto size = e.first;
			auto& t = e.second;

			Eigen::MatrixXd values(t.real() * (param.transpose()));
			values.conservativeResize(size, size);

			tableVals.push_back(
				std::move(values)
			);
		}

		double minEig = DBL_MAX;
		for(auto& t : tableVals)
		{
			Eigen::EigenSolver<Eigen::MatrixXd> solver(t);
			double localMin = solver.eigenvalues().real().minCoeff();
			if(localMin < minEig)
				minEig = localMin;
		}

		return minEig;
	}

	void QuadConstraint(QuadraticSolution& cons, Eigen::RowVectorXd& param,
		Eigen::RowVectorXd* val, Eigen::MatrixXd* grad, bool computeGrad = false)
	{
		// NOP: number of operator pairs
		// NV: number of variables
		// NC: number of constraints

		// param: (1, NV)

		auto& param1 = cons.Param1(); // (NOP, NV)
		auto& param2 = cons.Param2(); // (NOP, NV)
		auto& quad = cons.MatrixQuad(); // (NC, NOP)
		auto& line = cons.MatrixLine(); // (NC, NV)

		auto p1 = param1 * (param.transpose()); // (NOP, 1)
		auto p2 = param2 * (param.transpose()); // (NOP, 1)

		auto qconst = quad * (p1.cwiseProduct(p2)); // (NC, 1)
		auto lconst = line * (param.transpose()); // (NC, 1)

		*val = (qconst + lconst).real().transpose(); // (1, NC)

		if(!computeGrad)
			return;

		auto pgrad1 = param1.cwiseProduct(p2.replicate(1, param1.cols())); // (NOP, NV)
		auto pgrad2 = param2.cwiseProduct(p1.replicate(1, param2.cols())); // (NOP, NV)

		*grad = (line + (quad * (pgrad1 + pgrad2))).real(); // (NC, NV)
	}

	class SDPInit
	{
	private:
		std::vector<std::pair<size_t, Eigen::MatrixXcd>>* tables;
		Eigen::MatrixXd* A;
		Eigen::VectorXd* b;
		Eigen::VectorXd* init;
		double reg;
		int maxIters;
		double eps;

		int n;
		nlopt::opt opt;

		int cnt = 0;

	public:
		SDPInit(
			const std::vector<std::pair<size_t, Eigen::MatrixXcd>>& tables,
			const Eigen::MatrixXd& A,
			const Eigen::VectorXd& b,
			const Eigen::VectorXd& init,
			double lowerBound = -100.0,
			double upperBound = 100.0,
			double reg = 1.0,
			int maxIters = 5000,
			double eps = 1e-4
		)
			: reg(reg), maxIters(maxIters), eps(eps)
		{
			this->tables = const_cast<std::vector<std::pair<size_t, Eigen::MatrixXcd>>*>(&tables);
			this->A = const_cast<Eigen::MatrixXd*>(&A);
			this->b = const_cast<Eigen::VectorXd*>(&b);
			this->init = const_cast<Eigen::VectorXd*>(&init);

			assert(this->A->rows() == this->b->rows());
			assert(this->A->cols() == this->init->rows());

			n = init.rows();

			opt = nlopt::opt(nlopt::LN_COBYLA, n);
			opt.set_xtol_rel(eps);
			opt.set_ftol_rel(eps);
			opt.set_maxeval(maxIters);

			opt.set_lower_bounds(lowerBound);
			opt.set_upper_bounds(upperBound);

			opt.add_inequality_constraint(SDPInit::PCWrapper, (void*)this, eps);
			opt.set_min_objective(SDPInit::MinWrapper, (void*)this);
		}

		Eigen::VectorXd Solve()
		{
			std::vector<double> param(n);
			for(Eigen::Index i = 0; i < n; i++)
			{
				param[i] = (*init)(i, 0);
			}
			double minf = 0.0;

			try
			{
				nlopt::result result = opt.optimize(param, minf);
				return Eigen::Map<Eigen::VectorXd>(param.data(), n, 1);
			}
			catch(std::exception& e)
			{
				std::cerr << "nlopt in SDPInit failed: " << e.what() << std::endl;
				throw e;
			}
		}

	private:
		double PositivityConstraint(const double* x)
		{
			double* px = const_cast<double*>(x);
			return -MinimalEigval(*tables, Eigen::Map<Eigen::RowVectorXd>(px, 1, n));
		}

		double Minimize(const double* param)
		{
			cnt++;
			double* pParam = const_cast<double*>(param);
			auto vParam = Eigen::Map<Eigen::VectorXd>(pParam, n, 1);

			Eigen::MatrixXd& A = *this->A;
			Eigen::VectorXd& b = *this->b;
			Eigen::VectorXd& init = *this->init;

			double value = (A * vParam - b).squaredNorm() + this->reg * (vParam - init).squaredNorm();

			if(cnt % 100 == 0)
			{
				std::cout << "INFO: SDPInit iter " << cnt 
					<< " current value: " << value << std::endl;
			}

			return value;
		}

		static double PCWrapper(unsigned n, const double* x, double* grad, void* data)
		{
			return (reinterpret_cast<SDPInit*>(data))->PositivityConstraint(x);
		}

		static double MinWrapper(unsigned n, const double* x, double* grad, void* data)
		{
			return (reinterpret_cast<SDPInit*>(data))->Minimize(x);
		}
	};
}