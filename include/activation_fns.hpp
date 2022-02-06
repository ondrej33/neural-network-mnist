#ifndef ACTIVATION_FNS_H
#define ACTIVATION_FNS_H

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
   
    /* Applies activation fn on every sample vector in batch */
    virtual void apply_activation(FloatMat& ) { }
    /* Applies derivation fn on every sample vector in batch */
    virtual void apply_derivation(FloatMat& ) { }

    /* Applies activation fn on every item in input vector */
    virtual void apply_activation(FloatVec& ) { }
    /* Applies derivation fn on every item in input vector */
    virtual void apply_derivation(FloatVec& ) { }

    virtual ~ActivationFunction() = default;
};

/* RELU function and its derivative */
float relu(float x) { return std::fmax(0., x); }
float relu_deriv(float x) { return (x > 0) ? 1. : 0.; }

/* Class representing RELU activation function */
class ReluFunction : public ActivationFunction
{
    /* Applies activation fn on every input of every sample matrix */
    void apply_function(FloatMat& batch_mat, std::function<float(float)> fn) const
    {
        for (int i = 0; i < batch_mat.row_num(); ++i) {
            for (int j = 0; j < batch_mat[0].size(); ++j) {
                batch_mat[i][j] = fn(batch_mat[i][j]);
            }
        }
    }

    /* Applies activation fn on every item of input vector */
    void apply_function(FloatVec& vec, std::function<float(float)> fn) const
    {
        for (int i = 0; i < vec.size(); ++i) {
            vec[i] = fn(vec[i]);
        }
    }

public: 
    ReluFunction() = default;
    
    // version for whole batch at once
    void apply_activation(FloatMat& batch_mat) override { apply_function(batch_mat, relu); }
    void apply_derivation(FloatMat& batch_mat) override { apply_function(batch_mat, relu_deriv); }

    // version for one input at time
    void apply_activation(FloatVec& vec) override { apply_function(vec, relu); }
    void apply_derivation(FloatVec& vec) override { apply_function(vec, relu_deriv); }
};


/* Class representing Softmax activation function */
class SoftmaxFunction : public ActivationFunction
{
public: 
    SoftmaxFunction() = default;
    
    // version for whole batch at once
    void apply_activation(FloatMat& batch_mat) override { 
        for (auto& vec : batch_mat) {
            float sum = 0.;
            for (int i = 0; i < vec.size(); ++i) {
                vec[i] = std::exp(vec[i]);
                sum += vec[i];
            }

            for (int i = 0; i < vec.size(); ++i) {
                vec[i] = vec[i] / sum;
            }
        }
    }

    // version for one input at time
    void apply_activation(FloatVec& vec) override { 
        float sum = 0.;
        for (int i = 0; i < vec.size(); ++i) {
            vec[i] = std::exp(vec[i]);
            sum += vec[i];
        }

        for (int i = 0; i < vec.size(); ++i) {
            vec[i] = vec[i] / sum;
        }
    }

    /**
     * We will NOT specify "apply_derivation" function
     * The softmax derivation will be computed together with CrossEnropy derivation
     * And the network class will handle this
     */
};

#endif //ACTIVATION_FNS_H