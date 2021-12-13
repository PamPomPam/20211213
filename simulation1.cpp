#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp>

#include <iostream>
#include <cassert>
#include <fstream>
#include <chrono>

#include "siqrd.hpp"
#include "integration.hpp"

namespace ublas = boost::numeric::ublas;


template <typename Precision>
void write_to_file(ublas::matrix<Precision> SIQRD, std::string filename, Precision T, int N) {
    assert(SIQRD.size1() == N+1);
    assert(SIQRD.size2() == 5);
    std::ofstream outputfile(filename);
    for (size_t i = 0; i < N + 1; ++i) {
        outputfile << i * T / N << ' ';
        for (size_t j = 0; j < 5; ++j) { outputfile << SIQRD(i, j) << ' ';}
        outputfile << std::endl;
    }
}


int main(int argc, char *argv[]) {
    typedef double Precision;

    assert(argc == 3);
    int N = std::stoi(argv[1]);
    Precision T = std::stof(argv[2]);
    Precision delta_t = T / N;


    ublas::vector<Precision> prev_x(5);
    prev_x <<= 100, 5, 0, 0, 0;
    ublas::vector<Precision> next_x(5);

    // beta, mu, gamma, alpha, delta
    ublas::vector<Precision> p(5);
    p <<= 0.5, 0, 0.2, 0.005, 0;
    //p <<= 10.0, 0.0, 10.0, 1.0, 0.0; // can be used to compare to fortran code


    /*siqrd::time_deriv<Precision> myderiv(p);
    std::ofstream outputfile("fwe_no_measures.txt");
    outputfile << 0 << ' ';
    for (size_t j = 0; j < 5; ++j) { outputfile << prev_x(j) << ' ';}
    for (size_t n = 0; n < N; ++n) {
        integration::forward_euler(prev_x, next_x, myderiv, delta_t);
        prev_x = next_x;
        outputfile << std::endl << (n+1) * delta_t << ' ';
        for (size_t j = 0; j < 5; ++j) { outputfile << next_x(j) << ' ';}
    }


    p(4) = 0.2;
    myderiv.change_p(p);
    siqrd::error_jacob<Precision> myjacob(p);
    std::ofstream outputfile2("bwe_quarantine.txt");
    outputfile2 << 0 << ' ';
    for (size_t j = 0; j < 5; ++j) { outputfile2 << prev_x(j) << ' ';}
    for (size_t n = 0; n < N; ++n) {
        integration::backward_euler(prev_x, next_x, myderiv, myjacob, delta_t);
        prev_x = next_x;
        outputfile2 << std::endl << (n+1) * delta_t << ' ';
        for (size_t j = 0; j < 5; ++j) { outputfile2 << next_x(j) << ' ';}
    }

    p(4) = 0.9;
    myderiv.change_p(p);
    std::ofstream outputfile3("heun_lockdown.txt");
    outputfile3 << 0 << ' ';
    for (size_t j = 0; j < 5; ++j) { outputfile3 << prev_x(j) << ' ';}
    for (size_t n = 0; n < N; ++n) {
        integration::forward_euler(prev_x, next_x, myderiv, delta_t);
        prev_x = next_x;
        outputfile3 << std::endl << (n+1) * delta_t << ' ';
        for (size_t j = 0; j < 5; ++j) { outputfile3 << next_x(j) << ' ';}
    }*/

    auto t_start = std::chrono::high_resolution_clock::now();
    siqrd::time_deriv<double> myderiv(p);
    siqrd::error_jacob<double> myjacob(p);

    for (int i = 0; i < 100; ++i) {
        for (int n = 0; n < N; n++)
        integration::backward_euler(prev_x, next_x, myderiv, myjacob, delta_t);
        prev_x = next_x;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << " - Execution time: " << std::chrono::duration<double>(t_end-t_start).count() << std::endl;
    
}