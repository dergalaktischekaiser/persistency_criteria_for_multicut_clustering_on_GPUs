#include <tuple>
#include <vector>

#include <thrust/device_vector.h>

#include "device_vector_initialiser.h"
#include "input_reader.h"
#include "PersistencyCriteriaSolver.h"

int main(int argc, char *argv[]) {
    std::string folder = "multicut_instance/";
    std::string filename(argv[1]);
    INPUT_DATA_TUPLE input_data = read_file(folder + filename);
    
    thrust::device_vector<int> from = device_int_vector_initializr(0, input_data);
    thrust::device_vector<int> to = device_int_vector_initializr(1, input_data);
    thrust::device_vector<float> weights = device_float_vector_initializr(2, input_data);

    if (validate_input(from, to, weights)) {
        PersistencyCriteriaSolver solver(from, to, weights);
        solver.execute();
        // uncomment the following functions, if you would like to get more execution data
        // not recommended, when the amount of edges is a significantly big number
        /*solver.display_input();
        solver.display_abs_weights_per_vertex();
        solver.display_persistency_criteria();
        solver.display_node_mapping();
        solver.display_contracted_graph();*/
        solver.compare();
    }
    return 0;
}
