#include <algorithm>    // std::reverse
#include <iostream>     // std::cout
#include <memory>       // std::unique_ptr
#include <fstream>      // std::ifstream
#include <sstream>      // std::stringstream

#include "../include/main.hpp"
#include "../include/neural_network.hpp"


// Loads inputs from given file, arranges by lines, one line must contain "nums_per_line" numbers
// TODO: maybe load them as >char< type? it is just numbers 0-255
std::vector<std::unique_ptr<DoubleVec>> get_inputs(std::string file_name, int nums_per_line) {
    std::ifstream infile(file_name);
    std::vector<std::unique_ptr<DoubleVec>> data;
    std::string line;
    while (std::getline(infile, line))
    {
        std::stringstream line_stream(line);
        auto vec_ptr = std::make_unique<DoubleVec>(nums_per_line);
        // input vectors look just like "8,0,220,44,...,26,2"
        for (double num; line_stream >> num;) {
            vec_ptr->push_back(num);    
            
            if (line_stream.peek() == ',') {
                line_stream.ignore();
            }
        }
        data.push_back(std::move(vec_ptr)); // unique_ptr must be moved
    }
    return data;
}

// Loads labels from given file
// TODO: maybe load them as >char< type? it is just numbers 0-9
std::unique_ptr<DoubleVec> get_labels(std::string file_name) {
    std::ifstream infile(file_name);
    std::unique_ptr<DoubleVec> data;
    // TODO: initialize vec size so that it does not have to allocate all the time
    auto vec_ptr = std::make_unique<DoubleVec>();
    for (double num; infile >> num;) {
        vec_ptr->push_back(num);    

        if (infile.peek() == '\n') {
            infile.ignore();
        }
    }
    return vec_ptr;
}


int main() {
    // Simple neural network:  [N00, N01] -> [N10, N11] -> [N20, N21]
    std::vector<int> topology{2,2,2};
    double learn_rate = 0.5;
    int num_epochs = 20;
    int batch_size = 4;
    NeuralNetwork nw(topology, learn_rate, num_epochs, batch_size);

    std::cout << "INITIAL_WEIGHTS:\n";
    nw.print_weights();

    // Try it on some inputs
    nw.feed_input(DoubleMat(std::vector<DoubleVec>{
        DoubleVec(std::vector<double>{1.0, 1.0}),
        DoubleVec(std::vector<double>{0.0, 0.0}),
        DoubleVec(std::vector<double>{0.0, 1.0}),
        DoubleVec(std::vector<double>{1.0, 1.0}),
    }));

    std::cout << "NEURONS for [(1,1), (0,0), (0,1)]:\n";
    nw.forward_pass();
    nw.print_neurons();

    return 0;
}