#pragma once

#include <Eigen/Dense>
#include <Eigen/QR>
#include <iostream>
#include <cassert>

namespace Bootstrap
{
	Eigen::MatrixXcd NullSpace(Eigen::MatrixXcd& mat)
	{
		auto qr = mat.transpose().colPivHouseholderQr();

		auto rank = qr.rank();
		Eigen::MatrixXcd q(qr.matrixQ());
		Eigen::MatrixXcd null(q.block(0, rank, q.rows(), q.cols() - rank));

		assert(null.rows() == mat.cols());

		return null;
	}

	Eigen::MatrixXcd NullSpace(Eigen::MatrixXcd&& mat)
	{
		return NullSpace(static_cast<Eigen::MatrixXcd&>(mat));
	}

	Eigen::MatrixXcd RowSpace(Eigen::MatrixXcd& mat)
	{
		auto qr = mat.transpose().colPivHouseholderQr();

		auto rank = qr.rank();
		Eigen::MatrixXcd q(qr.matrixQ());
		Eigen::MatrixXcd row(q.block(0, 0, q.rows(), rank).transpose());

		assert(row.cols() == mat.cols());

		return row;
	}


	Eigen::MatrixXcd RowSpace(Eigen::MatrixXcd&& mat)
	{
		return RowSpace(static_cast<Eigen::MatrixXcd&>(mat));
	}
}