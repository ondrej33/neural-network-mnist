#include <iostream>     // std::cout

#include "../include/main.hpp"
#include "../include/neural_network.hpp"

int main() {
    std::string train_data_file = "data/fashion_mnist_train_vectors.csv";
    std::string train_labels_file = "data/fashion_mnist_train_labels.csv";
    std::string test_data_file = "data/fashion_mnist_test_vectors.csv";
    std::string test_labels_file = "data/fashion_mnist_test_labels.csv";

    std::string train_outputs = "data/trainPredictions";
    std::string test_outputs = "data/actualTestPredictions";

    // Simple neural network:  [N00, N01] -> [N10, N11] -> [N20, N21]
    std::vector<int> topology{2,8,2};
    double learn_rate = 0.05;
    int num_epochs = 100;
    int batch_size = 4;    
    NeuralNetwork nw(topology, learn_rate, num_epochs, batch_size, 
        train_data_file, train_labels_file, train_outputs);

    //std::cout << "INITIAL_WEIGHTS:\n";
    //nw.print_weights();

    // Try it on some inputs
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
    //nw.forward_pass();
    nw.print_neurons();

    /*
    std::cout << "NEURONS for [(1,1), (0,0), (0,1), (1,1)]:\n";
    nw.forward_pass();
    nw.print_neurons();
    */
    return 0;
}