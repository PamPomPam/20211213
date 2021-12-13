#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/banded.hpp>

#include <iostream>
#include <cassert>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <chrono>

#include "integration.hpp"

namespace ublas = boost::numeric::ublas;

template <typename Precision>
void function_time_deriv(ublas::vector<Precision> const& x, ublas::vector<Precision> & dx ) {
    dx = x;
    Precision m = -0.1;
    auto f = [&m] (Precision const& element) -> Precision {m += 0.1; return -10 * std::pow(element - m, 3);};
    std::transform(dx.begin(), dx.end(), dx.begin(), f);
}



template <typename Precision>
struct functor_time_deriv {
    functor_time_deriv() : myvec{50} {
        std::iota(myvec.begin(), myvec.end(), 0);
        myvec *= 0.1;
        std::cout << myvec << std::endl;
    }

    template <typename V1, typename V2>
    void operator()(V1 const& x, V2 &  dx) const {
        dx = x - myvec;
        dx = - 10. * element_prod(element_prod(dx, dx), dx);
    }

    ublas::vector<Precision> myvec;
};



template <typename Precision>
struct functor_error_jacobian {
    functor_error_jacobian() : myvec{50} {
        std::iota(myvec.begin(), myvec.end(), 0);
        myvec *= 0.1;
        std::cout << myvec << std::endl;
    }

    template <typename V>
    void operator()(ublas::vector<Precision> const& x, Precision const& delta_t, ublas::diagonal_matrix<Precision> & jacob) const {
        Precision temp;
        for (size_t i = 0; i < 50; ++i) {
            for (size_t j = 0; j < 50; ++j) {
                if (i != j) {
                    jacob(i, j) = 0;
                } else {
                    jacob(i, j) = -30 * pow(x(i) - myvec(i), 2) * x(i) - 1;
                }
            }
        }
        //ublas::vector<Precision> dx = x - myvec;
        //dx = - 30. * delta_t * element_prod(element_prod(dx, dx), x) - ublas::scalar_vector<Precision>(50) ;
        //ublas::diagonal_matrix<Precision> jacobian(dx.size(), dx.data());
    }

    ublas::vector<Precision> myvec;
};





int main(int argc, char *argv[]) {
    typedef double Precision;
    assert(argc == 3);
    int N = std::stoi(argv[1]);
    double T = std::stod(argv[2]);
    Precision delta_t = T / N;


    ublas::vector<Precision> prev_x(50);
    std::iota(prev_x.begin(), prev_x.end(), 1);
    prev_x *= 0.01;
    ublas::vector<Precision> next_x(50);


    /*auto lambda_time_deriv = [] ( \
    ublas::matrix_row<ublas::matrix<double>> const& x, ublas::vector<double> & dx) -> void {
        for (size_t i = 0; i < 50; ++i) {
            dx(i) = - 10 * std::pow(x(i) - 0.1 * i, 3);
        }
    };*/


    //integration::heun_method(SIQRD, myderiv, T);
    //integration::heun_method(SIQRD, derivative5, T);
    //backward_euler(SIQRD, myderiv, myjacobian, T);

    functor_time_deriv<Precision> myderiv;
    functor_error_jacobian<Precision> myjacob;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 20; ++i) {
        for (int n = 0; n < N; n++){
            integration::backward_euler(prev_x, next_x, myderiv, myjacob, delta_t);
            prev_x = next_x;
        }
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << " - Execution time: " << std::chrono::duration<double>(t_end-t_start).count() << std::endl;
    


}