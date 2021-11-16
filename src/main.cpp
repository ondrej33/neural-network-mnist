#include <algorithm>    // std::reverse
#include <iostream>     // std::cout
#include <memory>       // std::unique_ptr
#include <fstream>      // std::ifstream
#include <sstream>      // std::stringstream

#include "../include/main.hpp"
#include "../include/neural_network.hpp"


/**
 * Prints values of all neurons, one layer a line
 * Output layer is printed at the top, input at the bottom
 */
void print_neurons(std::vector<DoubleVec> neurons)
{
    std::reverse(neurons.begin(), neurons.end());
    for (const auto& layer: neurons) {
        for (const auto& neuron: layer) {
            std::cout << neuron << " | ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

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

int main() {
    // Simple neural network:  [N00, N01] -> [N10, N11] -> N3
    NeuralNetwork nw(std::vector<int>{1,2,1}, sigmoid, 0.5);

    // Give it input, now it is just '1'
    nw.feed_input(DoubleVec(std::vector<double>{1.0}));

    std::cout << "WEIGHTS:\n";
    nw.print_weights();

    std::cout << "NEURONS:\n";
    print_neurons(nw.eval_network());

    return 0;
}