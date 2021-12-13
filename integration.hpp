#ifndef integration_hpp
#define integration_hpp

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <iostream>

namespace ublas = boost::numeric::ublas;


namespace integration {
    template <typename Precision, typename Time_derivative>
    void forward_euler(ublas::vector<Precision> const& prev_x, ublas::vector<Precision> & next_x, \
     Time_derivative const& time_deriv, Precision const& delta_t) {
        time_deriv(prev_x, next_x);
        next_x = prev_x + delta_t * next_x;
    }

    template <typename Precision, typename Time_derivative>
    void heun_method(ublas::vector<Precision> const& prev_x, ublas::vector<Precision> & next_x, \
     Time_derivative const& time_deriv, Precision const& delta_t) {
        ublas::vector<Precision> dx;

        time_deriv(prev_x, dx);
        time_deriv(prev_x + delta_t * dx, next_x);
        next_x = prev_x + delta_t / 2 * (dx + next_x);
    }

    template <typename Precision, typename Time_derivative, typename Error_Jacobian>
    void backward_euler(ublas::vector<Precision> const& prev_x, ublas::vector<Precision> & next_x, \
     Time_derivative const& time_deriv, Error_Jacobian const& error_jacob, Precision const& delta_t) {
        size_t K = prev_x.size();
        ublas::vector<Precision> dx(K);
        ublas::vector<Precision> guessed_x(K);
        ublas::vector<Precision> error(K);
        ublas::matrix<Precision> jacobian(K, K);
        ublas::permutation_matrix<size_t> pm(K);
        ublas::permutation_matrix<size_t> pm_copy(K);

        time_deriv(prev_x, dx);
        guessed_x = prev_x + delta_t * dx;
        time_deriv(guessed_x, dx);
        error = prev_x + delta_t * dx - guessed_x; 
        while (norm_2(error) / norm_2(guessed_x) > 1e-10) {
            error_jacob(guessed_x, delta_t, jacobian);

            pm = pm_copy;
            ublas::lu_factorize(jacobian, pm);
            ublas::lu_substitute(jacobian, pm, error);
            guessed_x -= error;
            time_deriv(guessed_x, dx);
            error = prev_x + delta_t * dx - guessed_x; 
        }
        next_x = guessed_x;
    
    }

}



#endif
