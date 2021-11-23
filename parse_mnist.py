train_path = "data/mnist_train.csv"
test_path = "data/mnist_test.csv"

new_test_vectors_path = "data/mnist_test_vectors.csv"
new_train_vectors_path = "data/mnist_train_vectors.csv"
new_test_labels_path = "data/mnist_test_labels.csv"
new_train_labels_path = "data/mnist_train_labels.csv"


with open(test_path, 'r') as rf:
    vecs = open(new_test_vectors_path, "w")
    labs = open(new_test_labels_path, "w")
    pred = rf.read().split()[1:]
    for line in pred:
        label = line[0]
        data = line[2:]
        vecs.write(data + "\n")
        labs.write(label + "\n")
        
