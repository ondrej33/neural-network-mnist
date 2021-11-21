#include <iostream>     // std::cout

#include "../include/main.hpp"
#include "../include/neural_network.hpp"

int main() {
    /*
    std::string train_data_file = "data/fashion_mnist_train_vectors.csv";
    std::string train_labels_file = "data/fashion_mnist_train_labels.csv";
    std::string test_data_file = "data/fashion_mnist_test_vectors.csv";
    std::string test_labels_file = "data/fashion_mnist_test_labels.csv";

    std::string train_outputs = "data/trainPredictions";
    std::string test_outputs = "data/actualTestPredictions";
    */

    std::string train_data_file = "/home/ohuvar/pv021-project/data/fashion_mnist_train_vectors.csv";
    std::string train_labels_file = "/home/ohuvar/pv021-project/data/fashion_mnist_train_labels.csv";
    std::string test_data_file = "/home/ohuvar/pv021-project/data/fashion_mnist_test_vectors.csv";
    std::string test_labels_file = "/home/ohuvar/pv021-project/data/fashion_mnist_test_labels.csv";

    std::string train_outputs = "/home/ohuvar/pv021-project/data/trainPredictions";
    std::string test_outputs = "/home/ohuvar/pv021-project/data/actualTestPredictions";

    // Simple neural network:  [N00, N01] -> [N10, N11] -> [N20, N21]
    std::vector<int> topology{784,500,10};
    double learn_rate = 0.001;
    int num_epochs = 200;
    int batch_size = 64;    
    double momentum = 0.8;
    int steps_learn_decay = 10;
    NeuralNetwork nw(topology, learn_rate, num_epochs, batch_size, momentum, steps_learn_decay,
        train_data_file, train_labels_file, train_outputs);

    nw.train_network();
    nw.test_network(test_data_file, test_outputs);

    /*
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
    nw.forward_pass();
    nw.print_neurons();
    */

    return 0;
}