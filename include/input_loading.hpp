#include <memory>       // std::unique_ptr
#include <fstream>      // std::ifstream
#include <sstream>      // std::stringstream
#include <tuple>

/* Loads inputs from given file, arranges by lines, one line must contain "nums_per_line" numbers
 * TODO: maybe load them as >char< type? they are just numbers 0-255
 */
std::vector<FloatVec> get_inputs(std::string file_name, int nums_per_line) {
    std::ifstream infile(file_name);
    std::vector<FloatVec> data;
    data.reserve(10000); // reserve some memory, so that it does not have to allocate all the time

    std::string line;
    while (std::getline(infile, line))
    {
        std::stringstream line_stream(line);
        FloatVec vec;
        vec.reserve(nums_per_line);  // reserve memory for one input vector

        // input vectors look just like "8,0,220,44,...,26,2"
        for (float num; line_stream >> num;) {
            vec.push_back(num);    
            
            if (line_stream.peek() == ',') {
                line_stream.ignore();
            }
        }
        data.push_back(vec);
    }
    return data;
}

/* Loads labels from given file
 * TODO: maybe load them as >char< type? it is just numbers 0-9 */
std::vector<int> get_labels(std::string file_name) {
    std::ifstream infile(file_name);
    std::vector<int> vec;
    vec.reserve(10000);  // reserve some memory, so that it does not have to allocate all the time

    for (int num; infile >> num;) {
        vec.push_back(num);    

        if (infile.peek() == '\n') {
            infile.ignore();
        }
    }
    return vec;
}

struct VecLabelPair
{
    FloatVec input_vec;
    int label;
};

/* Loads both input vectors and labels 
 * There must be same number of lines in both files */
std::vector<VecLabelPair> load_vectors_labels(
    std::string file_name_vectors, std::string file_name_labels, int nums_per_vector)
{
    std::ifstream vector_stream(file_name_vectors);
    std::ifstream label_stream(file_name_labels);
    std::vector<VecLabelPair> result;
    result.reserve(10000); // reserve some memory, so that it does not have to allocate all the time

    std::string line;
    while (std::getline(vector_stream, line))
    {
        VecLabelPair vec_label_pair;
        vec_label_pair.input_vec.reserve(nums_per_vector);  // reserve some memory for one line

        label_stream >> vec_label_pair.label;
        if (label_stream.peek() == '\n') {
            label_stream.ignore();
        }

        // input vectors look just like "8,0,220,44,...,26,2"
        std::stringstream line_stream(line);
        for (float num; line_stream >> num;) {
            vec_label_pair.input_vec.push_back(num);    
            
            if (line_stream.peek() == ',') {
                line_stream.ignore();
            }
        }
        result.push_back(vec_label_pair);
    }
    return result;
}
