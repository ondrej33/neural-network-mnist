# Feed forward Neural network from scratch

This is a feed-forward neural network, designed to work on the MNIST and FashionMNIST datasets, which consist of thousands of pictures - either of handwritten digits, or product pictures of clothing from the Zalando catalogue. The network achieves accuracy over 88% in a few minutes.

The code is written in C++ 17 without using any 3rd party library - everything is built from the scratch. The project contains own library for matrix and vertex operations.

Network is trained using batch version of SGD with Adam optimizer and learning rate decay.

This was a semestral project for the class PV021 Neural Networks at the FI MU.

To successfully run the program, you must add a *data* folder with actual train+test data and label CSV files. These are too large to be saved in github repo.
