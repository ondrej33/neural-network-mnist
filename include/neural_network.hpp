#include <vector>
#include <cassert>
#include <functional>
#include <random>
#include <memory>
#include <fstream>
#include <math.h>       // log, exp

#include "layer.hpp"
#include "input_loading.hpp"

/* Main class representing neural network */
class NeuralNetwork
{
    /* numbers of neurons for each layer */
    std::vector<int> _topology;

    /* Matrix representing current batch, each row is one input vector */
    DoubleMat _input_batch;

    /* vector of current labels */
    std::vector<int> _batch_labels;

    /* vector of hidden and output layers */
    std::vector<std::unique_ptr<Layer>> _layers;

    /* training vectors and their labels */
    std::unique_ptr<std::vector<std::unique_ptr<VecLabelPair>>> _train_data_ptr = nullptr;

    /* output file stream for TRAINING predictions */
    std::ofstream _training_output_file;

    /* hyperparameters */
    double _init_learn_rate;
    int _num_epochs;
    int _batch_size;
    double _momentum;
    int _steps_learn_decay;

    ReluFunction relu_fn = ReluFunction();
    SoftmaxFunction soft_fn = SoftmaxFunction();

public:
    /* Creates object, >>consumes given data and label pointer<< 
     * TODO: uncomment data loading, for now it just takes too long */
    NeuralNetwork(std::vector<int> layer_sizes, double learn_rate, int num_epochs, 
        int batch_size, double momentum, int steps_learn_decay,
        std::string train_vectors, std::string train_labels, std::string train_output)
            : _topology(layer_sizes), 
              _input_batch(batch_size, layer_sizes[0]),
              _batch_labels(batch_size),
              _init_learn_rate(learn_rate),
              _num_epochs(num_epochs),
              _batch_size(batch_size),
              _momentum(momentum),
              _steps_learn_decay(steps_learn_decay)
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

        // load the train data
        load_train_data(train_vectors, train_labels, train_output);
    }

    /* number of layers excluding input */
    int layers_num() const { return _layers.size(); }

    int classes_num() const { return _topology[_topology.size() - 1]; }

    /* Puts new data and labels instead of old ones */
    void load_train_data(std::string vector_file, std::string label_file, std::string output)
    {
        _train_data_ptr = std::move(load_vectors_labels(vector_file, label_file, _topology[0]));
        _training_output_file.open(output);
    }

    /* Get new input batch and input labels */
    // TODO - change this for batch loading
    void feed_input(DoubleMat input_batch, std::vector<int> target_labels)
    {
        assert(input_batch.col_num() == _topology[0] && input_batch.row_num() == _batch_size);
        assert(target_labels.size() == _batch_size);
        _input_batch = input_batch;
        _batch_labels = target_labels;
    }

    /* Evaluate all neuron layers bottom-up (from input layer to output) */
    void forward_pass()
    {
        // initial layer is input, so lets use it to initiate first
        _layers[0]->forward(_input_batch);

        // now all other layers
        for (int i = 1; i < layers_num(); ++i) {
            _layers[i]->forward(_layers[i - 1]->_output_values);
        }
    }

    /* Calculates loss for the whole batch from outputs & true targets */
    double calculate_loss_cross_enthropy() 
    {
        // only care about output neurons with index same as LABEL for given input
        // we can ignore others, they would be 0 in hot-1-coded vectors
        double sum = 0.;
        for (int i = 0; i < _batch_size; ++i) {
            double correct_val_from_vector = _layers[layers_num() - 1]->_output_values[i][_batch_labels[i]];
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

    /* Executes backward pass on the last layer, which is kinda special
     * It involves both softmax and loss 
     * TODO - combine this with layer.backward_hidden - partly similar functionality*/
    void backward_pass_last_layer()
    {
        // lets start by computing derivations of "Softmax and CrossEntropy" wrt. Softmax inputs
        // thats easy, we just need those outputs and target labels
        DoubleMat softmax_outputs = _layers[layers_num() - 1]->_output_values;
        
        // and subtract 1 on indices of true target
        for (int i = 0; i < _batch_size; ++i) {
            softmax_outputs[i][_batch_labels[i]] -= 1; // we have derivatives wrt. inner pot
        }

        // TODO: normalize that computed gradient? - might not be good idea
        /*
        for (int i = 0; i < _batch_size; ++i) {
            softmax_outputs[i] /= _batch_size; // we have derivatives wrt. inner pot
        }
        */
        
        // just an alias for easier understanding
        DoubleMat& received_vals = softmax_outputs;
        
        const DoubleMat& outputs_prev_layer = (layers_num() == 1) ? _input_batch : _layers[layers_num() - 2]->_output_values;
        _layers[layers_num() - 1]->_deriv_weights = outputs_prev_layer.transpose() * received_vals;

        // for bias derivs, we just sum through the samples
        for (int i = 0; i < _batch_size; ++i) {
            for (int j = 0; j < _layers[layers_num() - 1]->_biases.size(); ++j) {
                _layers[layers_num() - 1]->_deriv_biases[j] += received_vals[i][j];
            }
        }

        // for derivation wrt. inputs, we multiply received values through the weigths (transponed to make it ok)
        _layers[layers_num() - 1]->_deriv_inputs = received_vals * _layers[layers_num() - 1]->_weights_in.transpose();
    }

    
    /* Using backpropagation, compute gradients wrt. inputs, weights and biases */
    void backward_pass()
    {
        // do the output layer separately (both softmax and cross entropy together)
        backward_pass_last_layer();

        if (layers_num() == 1) return;

        // now all other hidden layers except first, which is also different
        for (int i = layers_num() - 2; i > 0; --i) {
            _layers[i]->backward_hidden(_layers[i + 1]->_deriv_inputs, _layers[i - 1]->_output_values);
        }
        // first hidden layer takes directly inputs
        _layers[0]->backward_hidden(_layers[1]->_deriv_inputs, _input_batch);
    }

    /* Update weights and biases using previously computed gradients 
     * Uses momentum + learn rate which decays */
    void update_weights_biases(double learn_rate)
    {
        for (int i = 0; i < layers_num(); ++i) {
            auto weight_change = (_momentum * _layers[i]->_momentum_weights) - (learn_rate * _layers[i]->_deriv_weights);
            auto bias_change = (_momentum * _layers[i]->_momentum_biases) - (learn_rate * _layers[i]->_deriv_biases);
            _layers[i]->_weights_in += weight_change;
            _layers[i]->_biases += bias_change;

            // update momentums
            _layers[i]->_momentum_weights = std::move(weight_change);
            _layers[i]->_momentum_biases = std::move(bias_change);
        }
    }

    /* TODO */
    void one_epoch(double learn_rate)
    {
        /** TODO:
         * already has set batch of inputs + labels (labels are sparse)
         * do a forward pass with this batch - compute potentials+outputs for each layer (for all batch items)
         * backpropagate
         *    compute derivations wrt. outputs (last layer ez, other use derivations)
         *    compute derivations wrt. weights and biases -> gradient
         * change weights according to the gradient (subtract learn_rate * gradient)
         *    and also add momentum - gradient in prev step, multiplied by some alpha from [0,1]
         * 2 options - either implement learn rate decay (probably expnential?)
         *           - or use RMSprop instead of it - individually adapting learning rate for each weight (computed from gradient)
         * add weight decay + dropout
         */
        forward_pass();
        backward_pass();
        update_weights_biases(learn_rate);
    }

    /* TODO */
    void train_network()
    {
        // Randomly shuffle (both vectors+labels) and then take batches sequentially
        std::random_shuffle (_train_data_ptr->begin(), _train_data_ptr->end());
        int num_examples = _train_data_ptr->size();
             
        for (int i = 0; i < _num_epochs; i++) {
            // for every epoch choose batch of inputs+labels to train on
            for (int j = 0; j < _batch_size; ++j) {
                VecLabelPair& pair = *(*_train_data_ptr)[(i * _batch_size + j) % num_examples];
                _input_batch[j] = pair.input_vec / 255; // normalize inputs
                _batch_labels[j] = pair.label;
            }
 
            double learn_rate = _init_learn_rate / (1. + static_cast<double>(i) / _steps_learn_decay);
            one_epoch(learn_rate);

            print_batch_accuracy();
            std::cout << "loss: " << calculate_loss_cross_enthropy() << "\n";
        }

        // TODO evaluate train vectors and get rid of training values (we can move the ptr now)
        //predict_labels_to_file(_training_output_file, std::move(_train_data_ptr));
        _train_data_ptr = nullptr;
    }

    /* Checks the network output and returns percentage of correctly labeled samples */
    void print_batch_accuracy()
    {
        int correct = 0;
        // find the output label (largest of the softmax values)
        for (int sample_num = 0; sample_num < _batch_size; ++sample_num) {
            int label = 0;
            double largest = 0.;
            for (int i = 0; i < classes_num(); ++i) {
                double label_i_prob = _layers[layers_num() - 1]->_output_values[sample_num][i];
                if (label_i_prob > largest) {
                    largest = label_i_prob;
                    label = i;
                }
            }
            if (label == _batch_labels[sample_num]) {
                correct++;
            }
        }
        std::cout << correct << "/" << _batch_size << " ,"; // after this, loss is printed
    }

    /* Does one forward pass, gets the predicted label for given input vector */
    int predict_one_label(DoubleVec input_vec)
    {
        // normalize inputs
        input_vec /= 255.;

        // initial layer is input, so lets use it to initiate first
        _layers[0]->forward(DoubleMat(std::vector<DoubleVec>{input_vec}));

        // now all other layers
        for (int i = 1; i < layers_num(); ++i) {
            _layers[i]->forward(_layers[i - 1]->_output_values);
        }
        assert(_layers[layers_num() - 1]->_output_values.row_num() == 1); // only one sample in "batch"

        // find the output label (largest of the softmax values)
        int label = 0;
        double largest = 0.;
        for (int i = 0; i < classes_num(); ++i) {
            double label_i_prob = _layers[layers_num() - 1]->_output_values[0][i];
            if (label_i_prob > largest) {
                largest = label_i_prob;
                label = i;
            }
        }
        return label;
    }

    /* Prints label prediction for every input vector to given file */
    void predict_labels_to_file(std::ofstream& file, 
        std::unique_ptr<std::vector<std::unique_ptr<DoubleVec>>> input_vectors)
    {
        for (int i = 0; i < input_vectors->size(); ++i) {
            DoubleVec& input_vec = *(*input_vectors)[i];
            int label = predict_one_label(input_vec);
            file << label << "\n";
        }
    }

    /* Prints label prediction for every input vector to given file
     * similar as function above, but works for VecLabelPair that we use */
    void predict_labels_to_file(std::ofstream& file, 
        std::unique_ptr<std::vector<std::unique_ptr<VecLabelPair>>> input_data)
    {
        for (int i = 0; i < input_data->size(); ++i) {
            auto& vec = (*input_data)[i]->input_vec;
            int label = predict_one_label(vec);
            file << label << "\n";
        }
    }

    void test_network(std::string vector_file, std::string output)
    {
        auto test_vectors = get_inputs(vector_file, _topology[0]);
        std::ofstream output_file;
        output_file.open(output);
        predict_labels_to_file(output_file, std::move(test_vectors));
    }

    /* Prints values of all weights, one layer a line
     * output layer is printed at the top, input at the bottom */
    void print_weights() const
    {
        std::cout << std::setprecision(4) << std::fixed;
        for (int i = layers_num() - 1; i >= 0; --i) {
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
        for (int i = layers_num() - 1; i >= 0; --i) {
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

    ~NeuralNetwork() {
        if (_training_output_file.is_open()) { _training_output_file.close(); }
    }
};
