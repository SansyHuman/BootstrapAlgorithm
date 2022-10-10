#pragma once

#include <Eigen/Dense>
#include <Eigen/EigenValues>
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
			std::cout << solver.info() << std::endl;
			std::cout << solver.eigenvalues() << std::endl;
			double localMin = solver.eigenvalues().real().minCoeff();
			if(localMin < minEig)
				minEig = localMin;
		}

		return minEig;
	}
}