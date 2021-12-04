#ifndef LAYER_H
#define LAYER_H


#include <vector>
#include <cassert>
#include <functional>
#include <iostream>

#include "activation_fns.hpp"

/* Class representing one DENSE layer in the network */
template<int batch_size>
struct Layer
{
    /* Matrix of incoming weights, ith row contains ith weight for >every< neuron 
     * we choose this shape, so that we wont have to transpose every time during forward pass*/
    FloatMat _weights_in;

    /* Biases of each neuron */
    FloatVec _biases;

    /* Matrix where each row contains >inner potential of each neuron, for one input vector of the batch< */
    FloatMat _inner_potential;

    /* Matrix where each row contains >output value of each neuron, for one input vector of the batch< */
    // TODO: do we need to store this for RELU layers? - it is easily computable
    FloatMat _output_values;

    /* Gradient wrt. weights, biases, inputs */
    FloatMat _deriv_weights;
    FloatVec _deriv_biases;
    FloatMat _deriv_inputs;

    /* Previous gradients / cached stuff used for Adam */
    FloatMat _momentum_weights;
    FloatVec _momentum_biases;
    FloatMat _cached_weights;
    FloatVec _cached_biases;

    /* Activation function object (involves both function and derivative) */
    ActivationFunction& _activation_fn;

    Layer(int num_neurons, int num_neurons_prev_layer, ActivationFunction& fn, std::default_random_engine &generator)
        : _weights_in(num_neurons_prev_layer, num_neurons),
          _biases(num_neurons),
          _inner_potential(batch_size, num_neurons),
          _output_values(batch_size, num_neurons),
          _deriv_weights(num_neurons_prev_layer, num_neurons),
          _deriv_biases(num_neurons),
          _deriv_inputs(batch_size, num_neurons_prev_layer),
          _momentum_weights(num_neurons_prev_layer, num_neurons),
          _momentum_biases(num_neurons),
          _cached_weights(num_neurons_prev_layer, num_neurons),
          _cached_biases(num_neurons),
          _activation_fn(fn)
    {
        // TODO: initialize biases? - zero init should be ok
        
        // initiate weights - we use some kind of Xavier initialization for now
        std::normal_distribution<float> distribution(0.0, 1.0 / num_neurons);  // values have to be 0.0 and 1.0 (not 0 and 1)

        for (int i = 0; i < _weights_in.row_num(); ++i) {
            for (int j = 0; j < _weights_in.col_num(); ++j) {
                _weights_in[i][j] = distribution(generator);
            }
        }
    }

    /* getters for vectors, might be useful */
    FloatMat get_outputs() const { return _output_values; }
    FloatMat get_weights() const { return _weights_in; }
    FloatVec get_biases() const { return _biases; }

    int num_neurons() const { return _output_values.col_num(); }

    /* Take matrix, where each row is input vector, together batch_size inputs
     * input vector contains outputs from prev layer and we use them to compute our outputs */
    void forward(const FloatMat& input_batch)
    {
        // first compute inner potential
        _inner_potential = (input_batch * _weights_in).add_vec_to_all_rows(_biases);

        // now compute output values 
        _output_values = _inner_potential;
        _activation_fn.apply_activation(_output_values);
    }

    /* Backward pass, receives deriv_inputs from the following layer
     * THIS WILL NOT BE USED FOR LAST LAYER
     * computes deriv_weights, deriv_biases, deriv_inputs 
     * (deriv_inputs are basically derivs wrt outputs of prev layer) */
    void backward_hidden(FloatMat deriv_inputs_next_layer, const FloatMat& outputs_prev_layer)
    {
        /* Relu deriv would place 0 where inner_potential (relu input) was leq than 0, 
         * then we would mult it with incoming matrix from next layer.
         * We can instead just zero the values from next layer on indices, where inner_potential <= 0 */
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_neurons(); ++j) {
                if (_inner_potential[i][j] <= 0) {
                    deriv_inputs_next_layer[i][j] = 0;
                }
            }
        }
        // just an alias for easier understanding
        FloatMat& received_vals = deriv_inputs_next_layer;
        
        // TODO: optimize
        _deriv_weights = outputs_prev_layer.transpose() * received_vals;

        // for bias derivs, we just sum through the samples
        _deriv_biases = FloatVec(num_neurons());
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < _biases.size(); ++j) {
                _deriv_biases[j] += received_vals[i][j];
            }
        }

        // for derivation wrt. inputs, we multiply received values through the weigths (transponed to make it ok)
        // TODO: optimize
        _deriv_inputs = received_vals * _weights_in.transpose();
    }
};

#endif //LAYER_H