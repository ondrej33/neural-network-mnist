#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <array>
#include <cassert>
#include <functional>
#include <random>
#include <memory>
#include <fstream>
#include <math.h>       // log, exp

#include "layer.hpp"
#include "input_loading.hpp"

/* Main class representing neural network 
 * topology = array with numbers of neurons for every layer
 */
template<int batch_size, int num_epochs, int layers_total>
class NeuralNetwork
{
    /* numbers of neurons for each layer */
    std::array<int, layers_total> _topology;

    /* Matrix representing current batch, each row is one input vector */
    FloatMat _input_batch;

    /* vector of labels for current batch */
    std::vector<int> _batch_labels;

    /* vector of hidden and output layers */
    std::vector<Layer<batch_size>> _layers;

    /* training vectors and their labels */
    std::vector<VecLabelPair> _train_data;
    /* training data copy for evaluation (we need not shuffled data to evaluate) */
    std::vector<VecLabelPair> _train_data_copy;

    /* output file stream for TRAINING predictions */
    std::ofstream _training_output_file;

    /* hyperparameters */
    double _init_learn_rate;
    int _steps_learn_decay;
    double _epsilon;
    double _beta1;
    double _beta2;

    /* Activation function objects */
    ReluFunction relu_fn = ReluFunction();
    SoftmaxFunction soft_fn = SoftmaxFunction();

public:
    /* Creates object from given params/hyperparams, and training data files */
    NeuralNetwork(std::array<int, layers_total> layer_sizes, double learn_rate,  
        int steps_learn_decay, double epsilon, double beta1, double beta2,
        std::string train_vectors, std::string train_labels, std::string train_output,
        std::default_random_engine& generator)
            : _topology(layer_sizes), 
              _input_batch(batch_size, layer_sizes[0]),
              _batch_labels(batch_size),
              _init_learn_rate(learn_rate),
              _steps_learn_decay(steps_learn_decay),
              _epsilon(epsilon),
              _beta1(beta1),
              _beta2(beta2)
    {
        assert(layers_total > 1);
        assert(layers_total == _topology.size());

        // we dont want to have explicit layer for inputs
        // and we will initiate last layer separately
        for (int i = 1; i < layers_total - 1; ++i) {
            _layers.push_back(Layer<batch_size>(_topology[i], _topology[i-1], relu_fn, generator));
        }
        // last layer will have soft_max function
        _layers.push_back(Layer<batch_size>(_topology[layers_total - 1], _topology[layers_total - 2], soft_fn, generator));

        // load the train data
        load_train_data(train_vectors, train_labels, train_output);
    }

    int classes_num() const { return _topology[layers_total - 1]; }

    /* Puts new data and labels instead of old ones */
    void load_train_data(std::string vector_file, std::string label_file, std::string output)
    {
        _train_data = std::move(load_vectors_labels(vector_file, label_file, _topology[0]));
        _train_data_copy = _train_data;
        _training_output_file.open(output);
    }

    /* Evaluate all neuron layers bottom-up (from input layer to output) */
    void forward_pass()
    {
        // initial layer is input, so lets use it to initiate first
        _layers[0].forward(_input_batch);

        // now all other layers
        for (int i = 1; i < layers_total - 1; ++i) {
            _layers[i].forward(_layers[i - 1]._output_values);
        }
    }

    /* Calculates loss for the whole batch from outputs & true targets */
    float calculate_loss_cross_enthropy() 
    {
        // only care about output neurons with index same as LABEL for given input
        // we can ignore others, they would be 0 in hot-1-coded vectors
        float sum = 0.;
        for (int i = 0; i < batch_size; ++i) {
            float correct_val_from_vector = _layers[layers_total - 2]._output_values[i][_batch_labels[i]];
            // check if dont have 0, otherwise give some small value (same for 1 for symmetry)
            if (correct_val_from_vector < 1.0e-6) {
                correct_val_from_vector = 1.0e-6;
            }
            else if (correct_val_from_vector > 0.999999) {
                correct_val_from_vector = 0.999999;
            }
            sum += -std::log(correct_val_from_vector);
        }
        // return the mean
        return sum / batch_size;
    }

    /* Executes backward pass on the last layer, which is kinda special
     * It involves both softmax and loss 
     * TODO - combine this with layer.backward_hidden - partly similar functionality*/
    void backward_pass_last_layer()
    {
        // lets start by computing derivations of "Softmax AND CrossEntropy" (together) wrt. Softmax inputs
        // thats easy, we just need those outputs and target labels
        FloatMat softmax_outputs = _layers[layers_total - 2]._output_values;
        
        // and subtract 1 on indices of true target
        for (int i = 0; i < batch_size; ++i) {
            softmax_outputs[i][_batch_labels[i]] -= 1; // we receive derivatives wrt. inner potential
        }

        // normalizing computed gradient? - probably NOT good idea, slows computation?
        /*
        for (int i = 0; i < batch_size; ++i) {
            softmax_outputs[i] /= batch_size; // we have derivatives wrt. inner pot
        }
        */
        
        // just an alias for easier understanding
        FloatMat& received_vals = softmax_outputs;
        
        const FloatMat& outputs_prev_layer = (layers_total == 2) ? _input_batch : _layers[layers_total - 3]._output_values;
        // TODO: optimize
        _layers[layers_total - 2]._deriv_weights = std::move(outputs_prev_layer.transpose() * received_vals);

        // for bias derivs, we just sum through the samples (first annulate values)
        _layers[layers_total - 2]._deriv_biases = std::move(FloatVec(_layers[layers_total - 2]._deriv_biases.size()));
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < _layers[layers_total - 2]._biases.size(); ++j) {
                _layers[layers_total - 2]._deriv_biases[j] += received_vals[i][j];
            }
        }

        // for derivation wrt. inputs, we multiply received values through the weigths (transponed to make it ok)
        // TODO: optimize
        _layers[layers_total - 2]._deriv_inputs = std::move(received_vals * _layers[layers_total - 2]._weights_in.transpose());
    }

    
    /* Using backpropagation, compute gradients wrt. inputs, weights and biases */
    void backward_pass()
    {
        // do the output layer separately (both softmax and cross entropy together)
        backward_pass_last_layer();

        if (layers_total == 2) return;

        // now all other hidden layers except first, which is also different
        for (int i = layers_total - 3; i > 0; --i) {
            _layers[i].backward_hidden(_layers[i + 1]._deriv_inputs, _layers[i - 1]._output_values);
        }
        // first hidden layer takes directly inputs
        _layers[0].backward_hidden(_layers[1]._deriv_inputs, _input_batch);
    }

    /* Update weights and biases using previously computed gradients 
     * At the moment uses Adam optimizer + somehow learn rate which decays */
    void update_weights_biases(double learn_rate, int iteration)
    {
        for (int i = 0; i < layers_total - 1; ++i) {
            // first update momentum using computed gradients
            _layers[i]._momentum_weights = std::move(_beta1 * _layers[i]._momentum_weights + (1 - _beta1) * _layers[i]._deriv_weights);
            _layers[i]._momentum_biases = std::move(_beta1 * _layers[i]._momentum_biases + (1 - _beta1) * _layers[i]._deriv_biases);

            // compute corrected momentum (without this, it would be biased in early iterations)
            float correction = 1. - std::pow(_beta1, iteration + 1);
            auto better_momentum_weights = std::move(_layers[i]._momentum_weights / correction);
            auto better_momentum_biases = std::move(_layers[i]._momentum_biases / correction);

            // also update cache with squared gradients
            _layers[i]._cached_weights = std::move(_beta2 * _layers[i]._cached_weights + (1 - _beta2) * square_inside(_layers[i]._deriv_weights));
            _layers[i]._cached_biases = std::move(_beta2 * _layers[i]._cached_biases + (1 - _beta2) * square_inside(_layers[i]._deriv_biases));

            // again compute corrected cache (without this, it would be biased in early iterations)
            float correction2 = 1. - std::pow(_beta2, iteration + 1);
            auto better_cached_weights = std::move(_layers[i]._cached_weights / correction2);
            auto better_cached_biases = std::move(_layers[i]._cached_biases / correction2);

            // finally update the parameters
            _layers[i]._weights_in += divide_by_items((-learn_rate * better_momentum_weights), (add_scalar_to_all_items(sqrt_inside(better_cached_weights), _epsilon)));
            _layers[i]._biases += divide_by_items((-learn_rate * better_momentum_biases), (add_scalar_to_all_items(sqrt_inside(better_cached_biases), _epsilon)));
        }
    }

    /* One batch of training process */
    void one_batch(double learn_rate, int iter)
    {
        forward_pass();                           // evaluate current batch
        backward_pass();                          // backpropagate to compute gradients
        update_weights_biases(learn_rate, iter);  // update parameters using gradients
    }

    /* Whole training process
     * Shuffles data, iterates for few epochs
     * for every epoch always iterates through examples using batches */
    void train_network(std::default_random_engine& generator)
    {
        int num_examples = static_cast<int>(_train_data.size());
        // we will ignore last few examples in every epoch (it is randomly shuffled, so its probably OK)
        int batches_total = num_examples / batch_size;
             
        for (int i = 0; i < num_epochs; i++) {
            // update decaying learn rate
            double learn_rate = _init_learn_rate / (1. + static_cast<double>(i) / _steps_learn_decay);

            for (int batch_num = 0; batch_num < batches_total; ++batch_num) {
                // Randomly shuffle (both vectors+labels) and then take batches sequentially
                std::shuffle (_train_data.begin(), _train_data.end(), generator);

                // extract the examples for current batch of inputs+labels from training data
                for (int j = 0; j < batch_size; ++j) {
                    VecLabelPair& pair = _train_data[(i * batch_size + j) % num_examples];
                    _input_batch[j] = pair.input_vec / 255; // normalize inputs
                    _batch_labels[j] = pair.label;
                }

                one_batch(learn_rate, i);

                // Uncomment if needed progress printing 
                /*
                // for epoch print every 100th batch info
                if (batch_num % 100 == 0) {
                    std::cout << "ep " << i << ", b " << batch_num << " ";
                    print_batch_accuracy();
                    std::cout << "loss: " << calculate_loss_cross_enthropy() << std::endl;
                }
                */
            }
            std::cout << "epoch " << i << " finished" << std::endl;
         }

        // evaluate train vectors and get rid of training values (we can move them now)
        std::cout << "Training finished. Writing all training predictions now." << std::endl;
        predict_labels_to_file(_training_output_file, std::move(_train_data_copy));
    }

    /* Checks the network output and returns percentage of correctly labeled samples */
    void print_batch_accuracy()
    {
        int correct = 0;
        // find the output label (largest of the softmax values)
        for (int sample_num = 0; sample_num < batch_size; ++sample_num) {
            int label = 0;
            float largest = 0.;
            for (int i = 0; i < classes_num(); ++i) {
                float label_i_prob = _layers[layers_total - 2]._output_values[sample_num][i];
                if (label_i_prob > largest) {
                    largest = label_i_prob;
                    label = i;
                }
            }
            if (label == _batch_labels[sample_num]) {
                correct++;
            }
        }
        std::cout << correct << "/" << batch_size << " ,"; // after this, loss is printed
    }

    /* Does one forward pass, gets the predicted label for given input vector */
    int predict_one_label(FloatVec input_vec)
    {
        // normalize inputs
        input_vec /= 255.;

        // initial layer is input, so lets use it to initiate first
        _layers[0].forward(FloatMat(std::vector<FloatVec>{input_vec}));

        // now all other layers
        for (int i = 1; i < layers_total - 1; ++i) {
            _layers[i].forward(_layers[i - 1]._output_values);
        }
        assert(_layers[layers_total - 2]._output_values.row_num() == 1); // only one sample in "batch"

        // find the output label (largest of the softmax values)
        int label = 0;
        float largest = 0.;
        for (int i = 0; i < classes_num(); ++i) {
            float label_i_prob = _layers[layers_total - 2]._output_values[0][i];
            if (label_i_prob > largest) {
                largest = label_i_prob;
                label = i;
            }
        }
        return label;
    }

    /* Prints label prediction for every input vector to given file */
    void predict_labels_to_file(std::ofstream& file, 
        std::vector<FloatVec> input_vectors)
    {
        for (size_t i = 0; i < input_vectors.size(); ++i) {
            FloatVec& input_vec = input_vectors[i];
            int label = predict_one_label(input_vec);
            file << label << "\n";
        }
    }

    /* Prints label prediction for every input vector to given file
     * similar as function above, but works for VecLabelPair that we use */
    void predict_labels_to_file(std::ofstream& file, 
        std::vector<VecLabelPair> input_data)
    {
        for (size_t i = 0; i < input_data.size(); ++i) {
            auto& vec = input_data[i].input_vec;
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
        for (int i = layers_total - 2; i >= 0; --i) {
            for (int j = 0; j < _layers[i]._weights_in.col_num(); ++j) {
                std::cout << "[ ";
                for (int k = 0; k < _layers[i]._weights_in.row_num(); ++k) {
                    std::cout << _layers[i]._weights_in[k][j] << " ";
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
        for (int i = layers_total - 2; i >= 0; --i) {
            for (const auto& neuron_vec: _layers[i].get_outputs()) {
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


#endif //NEURAL_NETWORK_H