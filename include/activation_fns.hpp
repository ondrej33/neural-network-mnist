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
   
    /* Applies activation fn on every sample vector */
    virtual void apply_activation(DoubleMat& batch_mat) { }

    /* Applies activation fn on every sample vector */
    virtual void apply_derivation(DoubleMat& batch_mat) { }

    virtual ~ActivationFunction() = default;
};

/* RELU function and its derivative */
float relu(float x) { return std::fmax(0., x); }
float relu_deriv(float x) { return (x > 0) ? 1. : 0.; }

/* Class representing RELU activation function */
class ReluFunction : public ActivationFunction
{
    /* Applies activation fn on every input of every sample vector */
    void apply_function(DoubleMat& batch_mat, std::function<float(float)> fn) const
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
            float sum = 0.;
            for (int i = 0; i < vec.size(); ++i) {
                exponents[i] = std::exp(vec[i]);
                sum += exponents[i];
            }

            for (int i = 0; i < vec.size(); ++i) {
                vec[i] = exponents[i] / sum;
            }
        }
    }

    /**
     * We will NOT specify "apply_derivation" function
     * Softmax derivation will be computed together with CrossEnropy derivation
     * And the network class will handle this
     */
};
