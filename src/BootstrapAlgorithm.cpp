// BootstrapAlgorithm.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <iomanip>
#include <nlopt.hpp>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/QR>

#include "Trace.h"
#include "Algebra.h"
#include "Solver.h"
#include "Optimize.h"

using namespace Bootstrap;

typedef struct {
    double a, b;
} my_constraint_data;

double myvfunc(const std::vector<double>& x, std::vector<double>& grad, void* my_func_data)
{
    return sqrt(x[1]);
}

double myvconstraint(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
    my_constraint_data* d = reinterpret_cast<my_constraint_data*>(data);
    double a = d->a, b = d->b;
    return ((a * x[0] + b) * (a * x[0] + b) * (a * x[0] + b) - x[1]);
}

int main()
{
    complex g = complex(2.0);

    Matrix basicBasis[2] = { 'P', 'X' };
    Eigen::Matrix2cd commutator;
    commutator << complex(0.0), complex(0.0, -1.0),
        complex(0.0, 1.0), complex(0.0);
    auto matInfo = MatrixInfo<2>(basicBasis, commutator);

    Matrix useBasis[2] = { 'A', 'B' };
    Eigen::Matrix2cd coefficients;
    coefficients << complex(1.0), complex(0.0, -1.0),
        complex(1.0), complex(0.0, 1.0);
    coefficients *= 0.5;
    matInfo.AddBasis(useBasis, coefficients);

    auto hamil = TraceOperator<2>(matInfo, {
        Trace(complex(1.0), "PP"), Trace(complex(1.0), "XX"), Trace(g, "XXXX")
        });
    std::cout << "H = " << hamil << std::endl;

    std::vector<Trace> gauge = {
        Trace(complex(1.0), "XP"),
        Trace(complex(-1.0), "PX"),
        Trace(complex(0.0, -1.0), "")
    };
    auto solver = Solver<2>(hamil, gauge, useBasis, [](const std::string& op) {
            std::string res = "";
            for(size_t i = 0; i < op.size(); i++)
            {
                if(op.at(i) == 'A')
                    res += 'B';
                else if(op.at(i) == 'B')
                    res += 'A';
                else
                    assert(false);
            }
            std::reverse(res.begin(), res.end());
            return res;
        }
    );

    std::vector<std::pair<size_t, Eigen::MatrixXcd>> tables;
    QuadraticSolution cons;

    solver.TableWithConstrinats(3,
        [](const std::string& s) { return (int)s.size() % 2; },
        &tables,
        &cons
    );

    auto hamilRewrite = hamil.Rewrite(useBasis);

    auto& sol = cons.Solution();
    auto vec = OperatorToVector(sol, hamilRewrite);
    auto unit = TraceOperator<2>(matInfo, { Trace(complex(1.0), "") });

    Eigen::MatrixXd AOp(0, sol.NumVariables());
    AOp.setZero();
    Eigen::VectorXd BOp(0, 1);
    BOp.setZero();

    AOp = VStack(AOp, OperatorToVector(sol, unit)).real();
    Eigen::VectorXcd tmp(1, 1);
    tmp << complex(1.0);
    BOp = VStack(BOp, tmp).real();

    Eigen::VectorXd init(sol.NumVariables(), 1);
    init.setConstant(0.0);

    SDPInit sdpInit(tables, AOp, BOp, init, -10.0, 10.0, 1.0, 1000, 1e-8);
    auto param = sdpInit.Solve();

    std::cout << param << std::endl;

    nlopt::opt localOpt(nlopt::LN_COBYLA, 2);
    localOpt.set_xtol_rel(1e-8);
    localOpt.set_maxeval(1000);

    nlopt::opt opt(nlopt::AUGLAG, 2);
    opt.set_local_optimizer(localOpt);

    std::vector<double> lb(2);
    lb[0] = -HUGE_VAL; lb[1] = 0;
    opt.set_lower_bounds(lb);
    opt.set_min_objective(myvfunc, NULL);
    my_constraint_data data[2] = { {2,0}, {-1,1} };
    opt.add_inequality_constraint(myvconstraint, &data[0], 1e-8);
    opt.add_inequality_constraint(myvconstraint, &data[1], 1e-8);
    opt.set_xtol_rel(1e-8);
    opt.set_maxeval(10000);
    std::vector<double> x(2);
    x[0] = 1.234; x[1] = 5.678;
    double minf;

    try {
        nlopt::result result = opt.optimize(x, minf);
        std::cout << "found minimum at f(" << x[0] << "," << x[1] << ") = "
            << std::setprecision(10) << minf << std::endl;
    }
    catch(std::exception& e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }
}

// 프로그램 실행: <Ctrl+F5> 또는 [디버그] > [디버깅하지 않고 시작] 메뉴
// 프로그램 디버그: <F5> 키 또는 [디버그] > [디버깅 시작] 메뉴

// 시작을 위한 팁: 
//   1. [솔루션 탐색기] 창을 사용하여 파일을 추가/관리합니다.
//   2. [팀 탐색기] 창을 사용하여 소스 제어에 연결합니다.
//   3. [출력] 창을 사용하여 빌드 출력 및 기타 메시지를 확인합니다.
//   4. [오류 목록] 창을 사용하여 오류를 봅니다.
//   5. [프로젝트] > [새 항목 추가]로 이동하여 새 코드 파일을 만들거나, [프로젝트] > [기존 항목 추가]로 이동하여 기존 코드 파일을 프로젝트에 추가합니다.
//   6. 나중에 이 프로젝트를 다시 열려면 [파일] > [열기] > [프로젝트]로 이동하고 .sln 파일을 선택합니다.
