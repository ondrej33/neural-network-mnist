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

    std::vector<int> topology{784,50,30,10};
    double learn_rate = 0.003;
    int num_epochs = 750;
    int batch_size = 128;    
    double momentum = 0.75;
    int steps_learn_decay = 25;
    NeuralNetwork nw(topology, learn_rate, num_epochs, batch_size, momentum, steps_learn_decay,
        train_data_file, train_labels_file, train_outputs);

    nw.train_network();
    nw.test_network(test_data_file, test_outputs);

    /*
    // Try XOR on some inputs
    nw.feed_input(DoubleMat(std::vector<DoubleVec>{
        DoubleVec(std::vector<double>{1.0, 1.0}),
        DoubleVec(std::vector<double>{0.0, 0.0}),
        DoubleVec(std::vector<double>{1.0, 0.0}),
        DoubleVec(std::vector<double>{0.0, 1.0}),
        }), 
        std::vector<int>{0,0,1,1}
    );
    nw.train_network();

    // Try it on some inputs
    nw.feed_input(DoubleMat(std::vector<DoubleVec>{
        DoubleVec(std::vector<double>{1.0, 0.0}),
        DoubleVec(std::vector<double>{0.0, 0.0}),
        DoubleVec(std::vector<double>{0.0, 0.0}),
        DoubleVec(std::vector<double>{0.0, 1.0}),
        }), 
        std::vector<int>{0,0,0,1}
    );
    std::cout << "NEURONS:\n";
    nw.forward_pass();
    nw.print_neurons();
    */

    std::cout << "Time: " << time(NULL) - t1 << "\n";

    return 0;
}