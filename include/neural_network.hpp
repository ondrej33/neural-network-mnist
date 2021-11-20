#include <vector>
#include <cassert>
#include <functional>
#include <random>
#include <memory>
#include <math.h>       // log, exp

#include "layer.hpp"
#include "input_loading.hpp"

/* Main class representing neural network */
class NeuralNetwork
{
    /* numbers of neurons for each layer */
    std::vector<int> _topology;

    /* Matrix representing batch, each row is one input vector */
    DoubleMat _input_batch;

    /* vector of current labels */
    std::vector<int> _input_labels;

    /* vector of hidden and output layers */
    std::vector<std::unique_ptr<Layer>> _layers;

    /* pointer to the data */
    std::unique_ptr<std::vector<std::unique_ptr<DoubleVec>>> _data_ptr = nullptr;

    /* pointer to the labels */
    std::unique_ptr<std::vector<int>> _labels_ptr = nullptr;

    double _learn_rate;
    int _num_epochs;
    int _batch_size;

    ReluFunction relu_fn = ReluFunction();
    SoftmaxFunction soft_fn = SoftmaxFunction();

public:
    /* Creates object, >>consumes given data and label pointer<< 
     * TODO: uncomment data loading, for now it just takes too long */
    NeuralNetwork(std::vector<int> layer_sizes, double learn_rate, int num_epochs, 
        int batch_size, std::string data_file, std::string label_file)
            : _topology(layer_sizes), 
              _input_batch(batch_size, layer_sizes[0]),
              //_data_ptr(std::move(get_inputs(data_file, batch_size))),
              //_labels_ptr(std::move(get_labels(label_file))),
              _learn_rate(learn_rate),
              _num_epochs(num_epochs),
              _batch_size(batch_size)
    {
        assert(_topology.size() > 1);

        // we dont want to have explicit layer for inputs
        // and we will initiate last layer separately
        for (int i = 1; i < _topology.size() - 1; ++i) {
            _layers.push_back(std::make_unique<Layer>(_batch_size, _topology[i], _topology[i-1], relu_fn));
        }
        // last layer will have soft_max function
        _layers.push_back(std::make_unique<Layer>(
            _batch_size, _topology[_topology.size() - 1], _topology[_topology.size()-2], soft_fn
        ));
    }

    /* Puts new data and labels instead of old ones */
    void change_data_and_labels(std::string data_file, std::string label_file)
    {
        _data_ptr = std::move(get_inputs(data_file, _batch_size));
        _labels_ptr = std::move(get_labels(label_file));
    }

    /* Get new input batch and input labels */
    // TODO - change this for batch loading
    void feed_input(DoubleMat input_batch, std::vector<int> input_labels)
    {
        assert(input_batch.col_num() == _topology[0] && input_batch.row_num() == _batch_size);
        _input_batch = input_batch;
        _input_labels = input_labels;
    }

    /* Evaluate all neuron layers bottom-up (from input layer to output) */
    void forward_pass()
    {
        // initial layer is input, so lets use it to initiate first
        _layers[0]->forward(_input_batch);

        // now all other layers
        for (int i = 1; i < _layers.size(); ++i) {
            _layers[i]->forward(_layers[i - 1]->_output_values);
        }
    }

    /* Calculates loss for the whole batch from outputs & true targets */
    double calculate_loss_enthropy() 
    {
        // only care about output neurons with index same as LABEL for given input
        // we can ignore others, they would be 0 in hot-1-coded vectors
        double sum = 0.;
        for (int i = 0; i < _batch_size; ++i) {
            double correct_val_from_vector = _layers[_layers.size() - 1]->_output_values[i][_input_labels[i]];
            // check if dont have 0, otherwise give some small value (same for 1 for symmetry)
            if (correct_val_from_vector < 1.0e-7) {
                correct_val_from_vector = 1.0e-7;
            }
            else if (correct_val_from_vector > 0.9999999) {
                correct_val_from_vector = 0.9999999;
            }
            sum += -std::log(correct_val_from_vector);
        }
        // return the mean
        return sum / _batch_size;
    }

    
    /* Compute derivations wrt. neuron values using backpropagation 
     * Compute also gradient - derivations wrt. weights and biases */
    /* TODO */
    void backward_pass_values()
    {

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

   /* TODO */
    void train_network()
    {
        // TODO: random shuffle inputs at beginning? - but also the ouputs, so that they still correpond
                // or choose N random indices for every batch, this sounds OK and EZ
        // TODO: also normalize inputs?

        for (int i = 0; i < _num_epochs; i++) {
            // TODO: for every epoch choose some random batch of inputs/labels to train on
                // feed them in the network using feed_input
            one_epoch();
        }
    }

   /* TODO */
    void one_epoch()
    {
        /** TODO:
         * receive batch of inputs + labels
         *    transform labels to one hot vectors? or at least use them like that
         * do a forward pass with this batch - compute potentials+outputs for each layer (for all batch items)
         * backpropagate
         *    compute derivations wrt. outputs (last layer ez, other use derivations)
         *    compute derivations wrt. weights -> gradient
         * change weights according to the gradient (subtract learn_rate * gradient)
         *    and also add momentum - gradient in prev step, multiplied by some alpha from [0,1]
         * 2 options - either implement learn rate decay (probably expnential?)
         *           - or use RMSprop instead of it - individually adapting learning rate for each weight (computed from gradient)
         */

    }


    /* Prints values of all weights, one layer a line
     * output layer is printed at the top, input at the bottom */
    void print_weights() const
    {
        std::cout << std::setprecision(4) << std::fixed;
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
        std::cout << std::setprecision(4) << std::fixed;
        for (int i = _layers.size() - 1; i >= 0; --i) {
            for (const auto& neuron_vec: _layers[i]->get_outputs()) {
                std::cout << "[";
                for (int j = 0; j < neuron_vec.size(); ++j) {
                    std::cout << neuron_vec[j];
                    if (j != neuron_vec.size() - 1) {
                        std:: cout << " | ";
                    }
                }
                std::cout << "]  ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};
