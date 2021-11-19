#include <algorithm>    // std::reverse
#include <iostream>     // std::cout
#include <memory>       // std::unique_ptr
#include <fstream>      // std::ifstream
#include <sstream>      // std::stringstream

#include "../include/main.hpp"
#include "../include/neural_network.hpp"


// Loads inputs from given file, arranges by lines, one line must contain "nums_per_line" numbers
// TODO: maybe load just some of the lines?
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
// TODO: maybe load just some of the lines? (same as for input loading)
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
    // Simple neural network:  [N00, N01] -> [N10, N11] -> N2
    NeuralNetwork nw(std::vector<int>{2,2,1}, 0.5);

    // Give it input, just two '1'
    nw.feed_input(DoubleVec(std::vector<double>{1.0, 1.0}));

    std::cout << "WEIGHTS:\n";
    nw.print_weights();

    std::cout << "NEURONS:\n";
    nw.forward_pass();
    nw.print_neurons();
    return 0;
}