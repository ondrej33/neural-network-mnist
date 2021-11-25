#include <vector>
#include <cassert>
#include <functional>
#include <random>

#include "activation_fns.hpp"

/**
 * FIRST DEPLETED VERSION OF NETWORK, IT IS LEFT JUST FOR REFERENCE
 */

/* Extracts weights that goes into given neuron, from a matrix of all layer weights */
FloatVec get_ingoing_weights(const std::vector<FloatVec>& weights_from_layer, int neuron_to)
{
    FloatVec weights_to(weights_from_layer.size());
    for (int i = 0; i < weights_to.size(); ++i) {
        weights_to[i] = weights_from_layer[i][neuron_to];
    }
    return weights_to;
}


/* Main class representing neural network */
class NeuralNetwork
{
    // numbers of neurons for each layer
    std::vector<int> _topology;
    FloatVec _input;

    // weights are organized by [layer_from][neuron_from][neuron_to], where neuron_to is in next layer
    std::vector<std::vector<FloatVec>> _weights; 

    float _bias_hidden = 0;
    float _bias_output = 0;

    std::function<float(float)> _activation_fn;
    double _learn_rate;

public:
    NeuralNetwork(
        std::vector<int> layer_sizes, 
        std::function<float(float)> activation_fn, 
        double learn_rate)
            : _topology(layer_sizes), 
              _activation_fn(activation_fn), 
              _weights(layer_sizes.size() - 1),
              _learn_rate(learn_rate)
    {
        // initiate weights - we use some kind of Xavier initialization for now
        std::default_random_engine generator;

        for (int i = 0; i < _weights.size(); ++i) {
            _weights[i].resize(layer_sizes[i], FloatVec(layer_sizes[i + 1]));
            std::normal_distribution<float> distribution(0.0, 1.0 / layer_sizes[i]);  // values have to be 0.0 and 1.0

            for (int j = 0; j < _weights[i].size(); ++j) {
                for (int k = 0; k < _weights[i][j].size(); ++k) {
                    _weights[i][j][k] = distribution(generator);
                }
            }
        }
    }

    // Get new input values
    void feed_input(FloatVec input_vec)
    {
        assert(input_vec.size() == _topology[0]);
        _input = input_vec;
    }

    // Evaluate all neuron layers bottom-up (from input layer to output) 
    // return all neuron values, neurons are arranged by layers
    std::vector<FloatVec> eval_network()
    {
        // neurons by layers
        std::vector<FloatVec> neuron_values(_topology.size());
        neuron_values[0] = _input;

        // zero-th layer is input, so start with first
        for (int i = 1; i < neuron_values.size(); ++i) {
            neuron_values[i] = FloatVec(_topology[i]);

            FloatVec& prev_layer_values = neuron_values[i - 1];

            for (int j = 0; j < neuron_values[i].size(); ++j) {
                bool is_last_layer = (i == neuron_values.size() - 1);
            
                float bias = (is_last_layer) ? _bias_output : _bias_hidden;
                float inner_potential = bias + prev_layer_values * get_ingoing_weights(_weights[i - 1], j);
                neuron_values[i][j] = _activation_fn(inner_potential);
            }
        }
        return neuron_values;
    }

    // Computes derivations of Error_k wrt. every neuron y_j using backpropagation
    // return derivation values for all neurons, arranged by layers
    std::vector<FloatVec> back_propagation_sigmoid(
        std::vector<FloatVec> neuron_values, FloatVec desired_output)
    {
        // create vectors to contain derivation values
        std::vector<FloatVec> derivations(_topology.size());
        int last_row_idx = _topology.size() - 1;

        // first assign values to last row, it is special case
        derivations[last_row_idx] = FloatVec(_topology[last_row_idx]);
        derivations[last_row_idx] = neuron_values[last_row_idx] - desired_output;

        // now compute all the other derivation, layer by layer backwards
        for (int i = last_row_idx - 1; i > 0; --i) {  // we use i > 0 cause we dont care about input neurons
            derivations[i] = FloatVec(_topology[i]);

            // compute deriv for all neurons in our layer
            for (int j = 0; j < _topology[i]; ++j) {
                // this works only when we have FULLY CONNECTED network and SIGMOIDAL function
                for (int r = 0; r < _topology[i + 1]; ++r) {
                    // for sigmoidal function, dS(ksi) = S(ksi)*(1-S(ksi)), and S(ksi) is neuron's value
                    derivations[i][j] += derivations[i + 1][r] * neuron_values[i + 1][r] * _weights[i][j][r];
                }
            }            
        }
        // lets assign derivations for input neurons to 0, just that there is something
        derivations[last_row_idx] = FloatVec(std::vector<float>(_topology[0], 0));
        return derivations;
    }

    // Compute derivations of Error_k wrt. every weight w_ji using known derivations wrt. neuron values
    // return derivation values for all weights, arranged by [layers][from][to]
    std::vector<std::vector<FloatVec>> get_weight_derivations_sigmoid(
        std::vector<FloatVec> neuron_values, std::vector<FloatVec> neuron_derivations)
    {
        // dW_ij = dY_j * dActiv(ksi_j) * Y_i
        std::vector<std::vector<FloatVec>> weight_derivations(_weights.size());

        for (int layer_from = 0; layer_from < _weights.size(); ++layer_from) {
            weight_derivations[layer_from] = std::vector<FloatVec>(_weights[layer_from].size());
            for (int i = 0; i < _weights[layer_from].size(); ++i) {
                weight_derivations[layer_from][i] = FloatVec(_weights[layer_from][i].size());
                for (int j = 0; j < _weights[i].size(); ++j) {
                    // for sigmoidal function, dS(ksi_j) = S(ksi_j)*(1-S(ksi_j)), and S(ksi_j) is neuron j's value
                    weight_derivations[layer_from][i][j] = \
                        neuron_derivations[layer_from + 1][j] * neuron_values[layer_from + 1][j] * neuron_values[layer_from][i];
                }
            }
        }
        return weight_derivations;
    }

    // Prints values of all weights, one layer a line, output layer is printed at the top, input at the bottom
    void print_weights() const
    {
        for (int i = _weights.size() - 1; i >= 0; --i) {
            const auto& layer = _weights[i];
            for (const auto& from: layer) {
                std::cout << "[ ";
                for (const auto& weight: from) {
                    std::cout << weight << " ";
                }
                std::cout << "] ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    
};
