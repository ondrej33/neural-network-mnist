#include <iostream>     // std::cout
#include <ctime>
#include <array>
#include <chrono>

#include "../include/main.hpp"
#include "../include/neural_network.hpp"

std::string train_data_file = "data/fashion_mnist_train_vectors.csv";
std::string train_labels_file = "data/fashion_mnist_train_labels.csv";
std::string test_data_file = "data/fashion_mnist_test_vectors.csv";
std::string test_labels_file = "data/fashion_mnist_test_labels.csv";

std::string train_outputs = "data/trainPredictions";
std::string test_outputs = "data/actualTestPredictions";

constexpr int input_size = 784;
constexpr int output_size = 10;

constexpr int num_networks = 3;

template<int batch_size, int num_epochs, int layers_total>
void test_networks(std::vector<NeuralNetwork<batch_size, num_epochs, layers_total>> &networks)
{
    /**
     * TODO: evaluate also training data, maybe just via 1 network
     */

    auto test_vectors = get_inputs(test_data_file, input_size);
    std::ofstream output_file;
    output_file.open(test_outputs);
    std::array<int, output_size> label_nums; // array to collect predicted labels for all networks

    for (int i = 0; i < test_vectors.size(); ++i) {
        // annulate labels
        for (int j = 0; j < output_size; ++j) {
            label_nums[j] = 0;
        }

        FloatVec& test_vec = test_vectors[i];

        // predict labels in parallel for all networks
#pragma omp parallel for num_threads(num_networks)
        for (int j = 0; j < num_networks; ++j) {
            int label_j = networks[j].predict_one_label(test_vec);
            label_nums[label_j]++;
        }

        int best_label = std::distance(label_nums.begin(),std::max_element(label_nums.begin(), label_nums.end()));
        std::cout << label_nums[best_label] << "\n";
        output_file << best_label << "\n";
    }
}


int main() {
    auto t1 = std::chrono::steady_clock::now();

    /** 
     * TODO: solve randomness issues - one generator, not every layer having its own with same seed
     */

    constexpr int batch_size = 64;    
    constexpr double epsilon = 1e-7;
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    constexpr int num_epochs = 10;
    constexpr int layers_total = 3;

    // initialize different networks to train
    constexpr std::array<int, layers_total> topology_1{input_size, 64, output_size};
    constexpr double learn_rate_1 = 0.003;
    constexpr int epochs_learn_decay_1 = 2;
    NeuralNetwork<batch_size, num_epochs, layers_total> nw1 (topology_1, learn_rate_1, epochs_learn_decay_1, 
        epsilon, beta1, beta2, train_data_file, train_labels_file, train_outputs);

    constexpr std::array<int, layers_total> topology_2{input_size, 50, output_size};
    constexpr double learn_rate_2 = 0.003;
    constexpr int epochs_learn_decay_2 = 2;
    NeuralNetwork<batch_size, num_epochs, layers_total> nw2 (topology_2, learn_rate_2, epochs_learn_decay_2, 
        epsilon, beta1, beta2, train_data_file, train_labels_file, train_outputs);

    constexpr std::array<int, layers_total> topology_3{input_size, 32, output_size};
    constexpr double learn_rate_3 = 0.003;
    constexpr int epochs_learn_decay_3 = 2;
    NeuralNetwork<batch_size, num_epochs, layers_total> nw3 (topology_3, learn_rate_3, epochs_learn_decay_3, 
        epsilon, beta1, beta2, train_data_file, train_labels_file, train_outputs);

    /**
     * TODO: maybe train like ~5 small networks (784-64-10 and similar sizes) with slightly different params
     * in parallel (using pragma), and at the end, for each input get results from each nw (in parallel)
     * and take the label which the most networks picked OR which had the best summed up softmax value
     */

    // add networks to one collection
    std::vector<NeuralNetwork<batch_size, num_epochs, layers_total>> networks;
    networks.push_back(std::move(nw1));
    networks.push_back(std::move(nw2));
    networks.push_back(std::move(nw3));

    assert(num_networks == networks.size());

    // train networks in parallel
#pragma omp parallel for num_threads(num_networks)
    for (int i = 0; i < num_networks; ++i) {
        networks[i].train_network();
    }

    // test networks
    test_networks<batch_size,num_epochs,layers_total>(networks);

    auto t2 = std::chrono::steady_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << "\n";
    return 0;
}
