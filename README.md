# Feed forward Neural network from scratch

This is a feed-forward neural network made from scratch in C++, designed to tackle MNIST and FashionMNIST datasets. The datasets consist of thousands of pictures - either of handwritten digits, or product pictures of clothing from the Zalando catalogue. The network achieves an accuracy of over 88% on FashionMNIST after just a few minutes of training.

The code is written in C++ 17 without using any 3rd party library - everything is built from scratch, including my own library for matrix and vertex operations.

The network is trained using a batch version of SGD with Adam optimizer and learning rate decay.

This was a semestral project for the class PV021 Neural Networks at the FI MU.

### Running the network

To successfully run the program (both training and inference), you must add a `data` folder with actual train+test data and label CSV files. The data itself is not included in this GitHub repo, you can download it for example [here](https://www.kaggle.com/datasets/zalando-research/fashionmnist).

Use the evaluator script in the `python_evaluator` directory to evaluate the results.
