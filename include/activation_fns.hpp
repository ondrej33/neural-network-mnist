#include <algorithm>    // std::max
#include <math.h>       // log, exp
#include <iostream>

#include "lingebra.hpp"


/* Class representing arbitrary activation function with both the func and its derivative 
 * this will be base in our hierarchy */
class ActivationFunction
{
public: 
    ActivationFunction() = default; 
   
    virtual void apply_activation(DoubleMat& batch_mat) { }

    virtual void apply_derivation(DoubleMat& batch_mat) { }

    virtual ~ActivationFunction() = default;

};


double relu(double x) { return std::fmax(0., x); }
double relu_deriv(double x) { return (x > 0) ? 1. : 0.; }


/* Class representing RELU activation function */
class ReluFunction : public ActivationFunction
{
    void apply_function(DoubleMat& batch_mat, std::function<double(double)> fn) const
    {
        for (int i = 0; i < batch_mat.row_num(); ++i) {
            for (int j = 0; j < batch_mat[0].size(); ++j) {
                batch_mat[i][j] = fn(batch_mat[i][j]);
            }
        }
    }

public: 
    ReluFunction() = default;
    
    void apply_activation(DoubleMat& batch_mat) override { apply_function(batch_mat, relu); }

    void apply_derivation(DoubleMat& batch_mat) override { apply_function(batch_mat, relu_deriv); }
};


/* Class representing Softmax activation function */
class SoftmaxFunction : public ActivationFunction
{
public: 
    SoftmaxFunction() = default;
    
    void apply_activation(DoubleMat& batch_mat) override { 
        for (auto& vec : batch_mat) {
            DoubleVec exponents(vec.size());
            double sum = 0.;
            for (int i = 0; i < vec.size(); ++i) {
                exponents[i] = std::exp(vec[i]);
                sum += exponents[i];
            }

            for (int i = 0; i < vec.size(); ++i) {
                vec[i] = exponents[i] / sum;
            }
        }
    }

    void apply_derivation(DoubleMat& batch_mat) override { 
        // TODO
    }
};



/* Other functions that might be useful */

double softplus(double x) { return std::log(1. + std::exp(x)); }

double sigmoid(double x) { return 1. / (1. + std::exp(-x)); }

double tanh(double x) 
{ 
    double pos = exp(x);
    double neg = exp(-x);
    return (pos - neg) / (pos + neg); 
}