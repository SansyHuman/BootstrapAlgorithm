// BootstrapAlgorithm.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <iomanip>
#include <nlopt.hpp>
#include <complex>
#include <Eigen/Dense>

#include "Trace.h"

typedef struct {
    double a, b;
} my_constraint_data;

double myvfunc(const std::vector<double>& x, std::vector<double>& grad, void* my_func_data)
{
    if(!grad.empty()) {
        grad[0] = 0.0;
        grad[1] = 0.5 / sqrt(x[1]);
    }
    return sqrt(x[1]);
}

double myvconstraint(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
    my_constraint_data* d = reinterpret_cast<my_constraint_data*>(data);
    double a = d->a, b = d->b;
    if(!grad.empty()) {
        grad[0] = 3 * a * (a * x[0] + b) * (a * x[0] + b);
        grad[1] = -1.0;
    }
    return ((a * x[0] + b) * (a * x[0] + b) * (a * x[0] + b) - x[1]);
}

int main()
{
    auto commutators = Eigen::Matrix4cd();
    commutators << complex(), complex(0.0, -1.0), complex(), complex(),
        complex(0.0, 1.0), complex(), complex(), complex(),
        complex(), complex(), complex(), complex(0.0, -1.0),
        complex(), complex(), complex(0.0, 1.0), complex();
    std::cout << commutators << std::endl;
    char basis[4] = { 'P', 'X', 'Q', 'Y' };
    MatrixInfo<4> info(basis, commutators);

    char basis2[4] = { 'A', 'B', 'C', 'D' };
    auto coeff = Eigen::Matrix4cd();
    coeff << complex(1.0, 0.0), complex(0.0, -1.0), complex(0.0, -1.0), complex(-1.0, 0.0),
        complex(1.0, 0.0), complex(0.0, 1.0), complex(0.0, 1.0), complex(-1.0, 0.0),
        complex(1.0, 0.0), complex(0.0, -1.0), complex(0.0, 1.0), complex(1.0, 0.0),
        complex(1.0, 0.0), complex(0.0, 1.0), complex(0.0, -1.0), complex(1.0, 0.0);
    coeff *= 0.5;
    info.AddBasis(basis2, coeff);
    auto coefB = info.GetCoefficients('B');
    auto commutator = info.Commutator('A', 'B');

    nlopt::opt opt(nlopt::LD_MMA, 2);
    std::vector<double> lb(2);
    lb[0] = -HUGE_VAL; lb[1] = 0;
    opt.set_lower_bounds(lb);
    opt.set_min_objective(myvfunc, NULL);
    my_constraint_data data[2] = { {2,0}, {-1,1} };
    opt.add_inequality_constraint(myvconstraint, &data[0], 1e-8);
    opt.add_inequality_constraint(myvconstraint, &data[1], 1e-8);
    opt.set_xtol_rel(1e-4);
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
