#include <algorithm>    // std::reverse
#include <iostream>     // std::cout

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

int main() {
    // Simple neural network:  N1 -> N2 -> N3
    NeuralNetwork nw(std::vector<int>{1,1,1}, sigmoid, 0.5);

    // Give it input '1'
    nw.feed_input(DoubleVec(std::vector<double>{1}));

    std::cout << "WEIGHTS:\n";
    nw.print_weights();

    std::cout << "NEURONS:\n";
    print_neurons(nw.eval_network());

    return 0;
}