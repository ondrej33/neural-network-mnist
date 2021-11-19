#include <vector>
#include <cassert>
#include <functional>
#include <iostream>

#include "activation_fns.hpp"

/* forward declaration */
class NeuralNetwork;

/* Class representing one DENSE layer in the network */
class Layer
{
    /* Matrix of incoming weights, ith row contains ith weight for >every< neuron 
     * we choose this shape, so that we wont have to transpose every time during forward pass*/
    DoubleMat _weights_in;

    /* Output values of each neuron */
    DoubleVec _output_values;

    /* Biases of each neuron */
    DoubleVec _biases;

    /* Activation function object (involves both function and derivative) */
    ActivationFunction& _activation_fn;

    friend NeuralNetwork;

public:
    Layer(int num_neurons, int num_neurons_prev_layer, ActivationFunction& fn)
        : _weights_in(num_neurons_prev_layer, num_neurons),
          _output_values(num_neurons),
          _biases(num_neurons),
          _activation_fn(fn)
    {
        // no need for initializing output values
        // TODO: initialize biases? how
        
        // initiate weights - we use some kind of Xavier initialization for now
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 1.0 / num_neurons);  // values have to be 0.0 and 1.0

        for (int i = 0; i < _weights_in.row_num(); ++i) {
            for (int j = 0; j < _weights_in.col_num(); ++j) {
                _weights_in[i][j] = distribution(generator);
            }
        }
    }

    /* getters */
    DoubleVec get_outputs() const { return _output_values; }
    DoubleMat get_weights() const { return _weights_in; }
    DoubleVec get_biases() const { return _biases; }


    /* Take vector of inputs (outputs from prev layer) and use them to compute outputs */
    void forward(const DoubleVec& input_vec)
    {
        // first compute inner potential, then apply activation fn
        _output_values = _biases + input_vec * _weights_in;
        _activation_fn.apply_activation(_output_values);
    }

    /* TODO */
    void backward(const DoubleVec& input_vec)
    {
        // TODO
    }

};