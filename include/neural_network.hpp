#include <vector>
#include <cassert>
#include <functional>
#include <random>
#include <memory>

#include "layer.hpp"


/* Main class representing neural network */
class NeuralNetwork
{
    /* numbers of neurons for each layer */
    std::vector<int> _topology;

    /* vector of current inputs (might be matrix someday) */
    DoubleVec _input;

    /* vector of hidden and output layers */
    std::vector<std::unique_ptr<Layer>> _layers;

    double _learn_rate;
    ReluFunction relu_fn = ReluFunction();
    SoftmaxFunction soft_fn = SoftmaxFunction();


public:
    NeuralNetwork(std::vector<int> layer_sizes, double learn_rate)
            : _topology(layer_sizes), 
              _input(layer_sizes[0]),
              _learn_rate(learn_rate)
    {
        assert(_topology.size() > 1);

        // we dont want to have explicit layer for inputs
        // and we will initiate last layer separately
        for (int i = 1; i < _topology.size() - 1; ++i) {
            _layers.push_back(std::make_unique<Layer>(_topology[i], _topology[i-1], relu_fn));
        }
        // last layer will have soft_max function
        _layers.push_back(std::make_unique<Layer>(
            _topology[_topology.size() - 1], _topology[_topology.size()-2], soft_fn
        ));
    }

    // Get new input value
    void feed_input(DoubleVec input_vec)
    {
        assert(input_vec.size() == _topology[0]);
        _input = input_vec;
    }

    // Evaluate all neuron layers bottom-up (from input layer to output) 
    void forward_pass()
    {
        // initial layer is input, so lets use it to initiate first
        _layers[0]->forward(_input);

        // now all other layers
        for (int i = 1; i < _layers.size(); ++i) {
            _layers[i]->forward(_layers[i - 1]->_output_values);
        }
    }

    /*
    // Computes derivations of Error_k wrt. every neuron y_j using backpropagation
    // return derivation values for all neurons, arranged by layers
    std::vector<DoubleVec> back_propagation_sigmoid(
        std::vector<DoubleVec> neuron_values, DoubleVec desired_output)
    {
        // create vectors to contain derivation values
        std::vector<DoubleVec> derivations(_topology.size());
        int last_row_idx = _topology.size() - 1;

        // first assign values to last row, it is special case
        derivations[last_row_idx] = DoubleVec(_topology[last_row_idx]);
        derivations[last_row_idx] = neuron_values[last_row_idx] - desired_output;

        // now compute all the other derivation, layer by layer backwards
        for (int i = last_row_idx - 1; i > 0; --i) {  // we use i > 0 cause we dont care about input neurons
            derivations[i] = DoubleVec(_topology[i]);

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
        derivations[last_row_idx] = DoubleVec(std::vector<double>(_topology[0], 0));
        return derivations;
    }

    // Compute derivations of Error_k wrt. every weight w_ji using known derivations wrt. neuron values
    // return derivation values for all weights, arranged by [layers][from][to]
    std::vector<std::vector<DoubleVec>> get_weight_derivations_sigmoid(
        std::vector<DoubleVec> neuron_values, std::vector<DoubleVec> neuron_derivations)
    {
        // dW_ij = dY_j * dActiv(ksi_j) * Y_i
        std::vector<std::vector<DoubleVec>> weight_derivations(_weights.size());

        for (int layer_from = 0; layer_from < _weights.size(); ++layer_from) {
            weight_derivations[layer_from] = std::vector<DoubleVec>(_weights[layer_from].size());
            for (int i = 0; i < _weights[layer_from].size(); ++i) {
                weight_derivations[layer_from][i] = DoubleVec(_weights[layer_from][i].size());
                for (int j = 0; j < _weights[i].size(); ++j) {
                    // for sigmoidal function, dS(ksi_j) = S(ksi_j)*(1-S(ksi_j)), and S(ksi_j) is neuron j's value
                    weight_derivations[layer_from][i][j] = \
                        neuron_derivations[layer_from + 1][j] * neuron_values[layer_from + 1][j] * neuron_values[layer_from][i];
                }
            }
        }
        return weight_derivations;
    }
    */

    // Prints values of all weights, one layer a line, output layer is printed at the top, input at the bottom
    void print_weights() const
    {
        for (int i = _layers.size() - 1; i >= 0; --i) {
            for (int j = 0; j < _layers[i]->_weights_in.col_num(); ++j) {
                std::cout << "[ ";
                for (int k = 0; k < _layers[i]->_weights_in.row_num(); ++k) {
                    std::cout << _layers[i]->_weights_in[k][j] << " ";
                }
                std::cout << "] ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    /**
     * Prints values of all neurons, one layer a line
     * Output layer is printed at the top, input at the bottom
     */
    void print_neurons()
    {
        for (int i = _layers.size() - 1; i >= 0; --i) {
            for (const auto& neuron: _layers[i]->get_outputs()) {
                std::cout << neuron << " | ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};
