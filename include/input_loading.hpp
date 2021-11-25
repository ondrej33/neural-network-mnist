#include <memory>       // std::unique_ptr
#include <fstream>      // std::ifstream
#include <sstream>      // std::stringstream
#include <tuple>

/* Loads inputs from given file, arranges by lines, one line must contain "nums_per_line" numbers
 * TODO: maybe load them as >char< type? they are just numbers 0-255
 */
std::unique_ptr<std::vector<std::unique_ptr<FloatVec>>> get_inputs(std::string file_name, int nums_per_line) {
    std::ifstream infile(file_name);
    auto data = std::make_unique<std::vector<std::unique_ptr<FloatVec>>>();
    std::string line;
    while (std::getline(infile, line))
    {
        std::stringstream line_stream(line);
        auto vec_ptr = std::make_unique<FloatVec>();
        // input vectors look just like "8,0,220,44,...,26,2"
        for (float num; line_stream >> num;) {
            vec_ptr->push_back(num);    
            
            if (line_stream.peek() == ',') {
                line_stream.ignore();
            }
        }
        data->push_back(std::move(vec_ptr)); // unique_ptr must be moved
    }
    return data;
}

/* Loads labels from given file
 * TODO: maybe load them as >char< type? it is just numbers 0-9 */
std::unique_ptr<std::vector<int>> get_labels(std::string file_name) {
    std::ifstream infile(file_name);
    auto vec_ptr = std::make_unique<std::vector<int>>();
    // reserve some memory, so that it does not have to allocate all the time
    vec_ptr->reserve(10000);
    for (int num; infile >> num;) {
        vec_ptr->push_back(num);    

        if (infile.peek() == '\n') {
            infile.ignore();
        }
    }
    return vec_ptr;
}

struct VecLabelPair
{
    FloatVec input_vec;
    int label;
};

/* Loads both input vectors and labels 
 * There must be same number of lines in both files */
std::unique_ptr<std::vector<std::unique_ptr<VecLabelPair>>> load_vectors_labels(
    std::string file_name_vectors, std::string file_name_labels, int nums_per_vector)
{
    std::ifstream vector_stream(file_name_vectors);
    std::ifstream label_stream(file_name_labels);
    auto result = std::make_unique<std::vector<std::unique_ptr<VecLabelPair>>>();
    // reserve some memory, so that it does not have to allocate all the time
    result->reserve(10000);
    std::string line;
    while (std::getline(vector_stream, line))
    {
        auto vec_label_ptr = std::make_unique<VecLabelPair>();
        // reserve some memory for one line
        vec_label_ptr->input_vec.reserve(784);

        label_stream >> vec_label_ptr->label;
        if (label_stream.peek() == '\n') {
            label_stream.ignore();
        }

        // input vectors look just like "8,0,220,44,...,26,2"
        std::stringstream line_stream(line);
        for (float num; line_stream >> num;) {
            vec_label_ptr->input_vec.push_back(num);    
            
            if (line_stream.peek() == ',') {
                line_stream.ignore();
            }
        }
        result->push_back(std::move(vec_label_ptr)); // unique_ptr must be moved
    }
    return result;
}