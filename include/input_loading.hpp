#include <memory>       // std::unique_ptr
#include <fstream>      // std::ifstream
#include <sstream>      // std::stringstream

/* Loads inputs from given file, arranges by lines, one line must contain "nums_per_line" numbers
 * TODO: maybe load them as >char< type? they are just numbers 0-255
 */
std::unique_ptr<std::vector<std::unique_ptr<DoubleVec>>> get_inputs(std::string file_name, int nums_per_line) {
    std::ifstream infile(file_name);
    auto data = std::make_unique<std::vector<std::unique_ptr<DoubleVec>>>();
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
        data->push_back(std::move(vec_ptr)); // unique_ptr must be moved
    }
    return data;
}

/* Loads labels from given file
 * TODO: maybe load them as >char< type? it is just numbers 0-9 */
std::unique_ptr<std::vector<int>> get_labels(std::string file_name) {
    std::ifstream infile(file_name);

    // TODO: initialize vec size so that it does not have to allocate all the time
    auto vec_ptr = std::make_unique<std::vector<int>>();
    for (int num; infile >> num;) {
        vec_ptr->push_back(num);    

        if (infile.peek() == '\n') {
            infile.ignore();
        }
    }
    return vec_ptr;
}