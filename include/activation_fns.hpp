#include <algorithm>    // std::max
#include <math.h>       // log, exp

/**
 * Some of the functions that can be used as the activation functions
 */


double relu(double x) { return std::fmax(0, x); }

double softplus(double x) { return std::log(1 + std::exp(x)); }

double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }

double tanh(double x) 
{ 
    double pos = exp(x);
    double neg = exp(-x);
    return (pos - neg) / (pos + neg); 
}