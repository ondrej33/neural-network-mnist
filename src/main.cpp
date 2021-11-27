#include <iostream>     // std::cout
#include <ctime>

#include "../include/main.hpp"
#include "../include/neural_network.hpp"

int main() {
    auto t1 = time(NULL);

    std::string train_data_file = "data/fashion_mnist_train_vectors.csv";
    std::string train_labels_file = "data/fashion_mnist_train_labels.csv";
    std::string test_data_file = "data/fashion_mnist_test_vectors.csv";
    std::string test_labels_file = "data/fashion_mnist_test_labels.csv";

    std::string train_outputs = "data/trainPredictions";
    std::string test_outputs = "data/actualTestPredictions";

    std::vector<int> topology{784,64,10};
    constexpr double learn_rate = 0.003;
    constexpr int num_epochs = 1;
    constexpr int batch_size = 64;    
    constexpr int epochs_learn_decay = 2;
    constexpr double epsilon = 1e-7;
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    NeuralNetwork nw(topology, learn_rate, num_epochs, batch_size, 
        epochs_learn_decay, epsilon, beta1, beta2,
        train_data_file, train_labels_file, train_outputs);

    nw.train_network();
    nw.test_network(test_data_file, test_outputs);

    std::cout << "Time: " << time(NULL) - t1 << "\n";

    return 0;
}
