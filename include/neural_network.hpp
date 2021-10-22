 #include <vector>
#include <cassert>
#include <functional>
#include <random>

#include "activation_fns.hpp"
#include "error_fns.hpp"


/* Extracts weights that goes into given neuron, from a matrix of all layer weights */
DoubleVec get_ingoing_weights(const std::vector<DoubleVec>& weights_from_layer, int neuron_to)
{
    DoubleVec weights_to(weights_from_layer.size());
    for (int i = 0; i < weights_to.size(); ++i) {
        weights_to[i] = weights_from_layer[i][neuron_to];
    }
    return weights_to;
}


/* Main class representing neural network */
class NeuralNetwork
{
    std::vector<int> _topology;
    DoubleVec _input;

    // weights are organized by [layer_from][neuron_from][neuron_to]
    std::vector<std::vector<DoubleVec>> _weights; 

    double _bias_hidden = 0;
    double _bias_output = 0;

    std::function<double(double)> _activation_fn;
    double _learn_rate;

public:
    NeuralNetwork(std::vector<int> layer_sizes, std::function<double(double)> fn, double learn_rate)
        : _topology(layer_sizes), _activation_fn(fn), _weights(layer_sizes.size() - 1), _learn_rate(learn_rate)
    {
        // initiate weights - we use some kind of Xavier initialization for now
        std::default_random_engine generator;

        for (int i = 0; i < _weights.size(); ++i) {
            _weights[i].resize(layer_sizes[i], DoubleVec(layer_sizes[i + 1]));
            std::normal_distribution<double> distribution(0, 1 / layer_sizes[i]);

            for (int j = 0; j < _weights[i].size(); ++j) {
                for (int k = 0; k < _weights[i][j].size(); ++k) {
                    _weights[i][j][k] = distribution(generator);
                }
            }
        }
    }

    // Get new input values
    void feed_input(DoubleVec input_vec)
    {
        assert(input_vec.size() == _topology[0]);
        _input = input_vec;
    }

    // Evaluate all neuron layers bottom-up (from input layer to output) and return all neuron values
    std::vector<DoubleVec> eval_network()
    {
        // neurons by layers
        std::vector<DoubleVec> neuron_values(_topology.size());
        neuron_values[0] = _input;

        // zero-th layer is input, so start with first
        for (int i = 1; i < neuron_values.size(); ++i) {
            neuron_values[i] = DoubleVec(_topology[i]);

            DoubleVec& prev_layer_values = neuron_values[i - 1];

            for (int j = 0; j < neuron_values[i].size(); ++j) {
                bool is_last_layer = (i == neuron_values.size() - 1);
            
                double bias = (is_last_layer) ? _bias_output : _bias_hidden;
                double inner_potential = bias + prev_layer_values * get_ingoing_weights(_weights[i - 1], j);
                neuron_values[i][j] = _activation_fn(inner_potential);
            }
        }
        return neuron_values;
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
