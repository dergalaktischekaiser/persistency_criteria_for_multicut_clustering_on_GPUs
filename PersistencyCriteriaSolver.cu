#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>

#include <chrono>
#include <fstream>
#include <thread>

#include "dCOO.h"
#include "device_vector_initialiser.h"
#include "PersistencyCriteriaSolver.h"

PersistencyCriteriaSolver::PersistencyCriteriaSolver(
            const thrust::device_vector<int> &_from,
            const thrust::device_vector<int> &_to,
            const thrust::device_vector<float> &_weights
        ) : from(_from), to(_to), weights(_weights) {}

void PersistencyCriteriaSolver::execute() {
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    dCOO dcoo(std::move(from), std::move(to), std::move(weights), true, false);
    weights = dcoo.get_data();
    from = dcoo.get_row_ids();
    to = dcoo.get_col_ids();

    auto m = from.size();
    thrust::device_vector<int> counter(m);
    for (int i = 0; i < m; ++i) counter[i] = i;
    thrust::device_vector<float> absolute_weights(m);
    

    thrust::device_vector<int> fromFirstCopy(m); thrust::device_vector<int> fromSecondCopy(m);
    thrust::copy(thrust::device, from.begin(), from.end(), fromFirstCopy.begin());    
    thrust::device_vector<int> toFirstCopy(m); thrust::device_vector<int> toSecondCopy(m);
    thrust::copy(thrust::device, to.begin(), to.end(), toFirstCopy.begin());    
    thrust::device_vector<float> weightsFirstCopy(m); thrust::device_vector<float> weightsSecondCopy(m);
    thrust::copy(thrust::device, weights.begin(), weights.end(), weightsFirstCopy.begin());

    thrust::sort_by_key(thrust::device, fromFirstCopy.begin(), fromFirstCopy.end(), weightsFirstCopy.begin());
    thrust::transform(thrust::device, weightsFirstCopy.begin(), weightsFirstCopy.end(), absolute_weights.begin(), abs_functor());
    thrust::device_vector<int> output_from(m);
    thrust::device_vector<float> output_weights(m);
    thrust::reduce_by_key(thrust::device, fromFirstCopy.begin(), fromFirstCopy.end(), absolute_weights.begin(), output_from.begin(), output_weights.begin());
    while (output_from[output_from.size() - 1] == 0) output_from.pop_back();
    output_from.resize(output_from.size());
    while (output_weights[output_weights.size() - 1] == 0) output_weights.pop_back();
    output_weights.resize(output_weights.size());

    thrust::copy(thrust::device, from.begin(), from.end(), fromSecondCopy.begin());
    thrust::copy(thrust::device, to.begin(), to.end(), toSecondCopy.begin());
    thrust::copy(thrust::device, weights.begin(), weights.end(), weightsSecondCopy.begin());

    thrust::sort_by_key(thrust::device, toSecondCopy.begin(), toSecondCopy.end(), weightsSecondCopy.begin());
    thrust::device_vector<int> output_to(m);
    thrust::device_vector<float> output_weights_2(m);
    thrust::transform(thrust::device, weightsSecondCopy.begin(), weightsSecondCopy.end(), absolute_weights.begin(), abs_functor());
    thrust::reduce_by_key(thrust::device, toSecondCopy.begin(), toSecondCopy.end(), absolute_weights.begin(), output_to.begin(), output_weights_2.begin());
    while (output_to[output_to.size() - 1] == 0) output_to.pop_back();
    output_to.resize(output_to.size());
    while (output_weights_2[output_weights_2.size() - 1] == 0) output_weights_2.pop_back();
    output_weights_2.resize(output_weights_2.size());   

    thrust::device_vector<int> all_nodes(output_from.size() + output_to.size());
    thrust::device_vector<float> abs_weights(output_weights.size() + output_weights_2.size());
    thrust::copy(thrust::device, output_from.begin(), output_from.end(), all_nodes.begin());
    thrust::copy(thrust::device, output_to.begin(), output_to.end(), all_nodes.begin() + output_from.size());
    thrust::copy(thrust::device, output_weights.begin(), output_weights.end(), abs_weights.begin());
    thrust::copy(thrust::device, output_weights_2.begin(), output_weights_2.end(), abs_weights.begin() + output_weights.size());

    thrust::sort_by_key(thrust::device, all_nodes.begin(), all_nodes.end(), abs_weights.begin());
    thrust::reduce_by_key(thrust::device, all_nodes.begin(), all_nodes.end(), abs_weights.begin(), all_nodes.begin(), abs_weights.begin());
    all_nodes_ = all_nodes;
    abs_weights_ = abs_weights;

    thrust::device_vector<int> PERSISTENCY_CRITERIA(m, 0);


    
    auto first_edge = thrust::make_zip_iterator(thrust::make_tuple(from.begin(), to.begin(), weights.begin(), counter.begin()));
    auto after_the_last_edge = thrust::make_zip_iterator(thrust::make_tuple(from.end(), to.end(), weights.end(), counter.end()));

    compute_per_crit(first_edge, after_the_last_edge, abs_weights, PERSISTENCY_CRITERIA);
    /*thrust::for_each(
        thrust::device,
        first_edge,
        after_the_last_edge,
        __PER__CRIT__FUNCTOR__<float, int>(
            thrust::raw_pointer_cast(abs_weights.data()), thrust::raw_pointer_cast(PERSISTENCY_CRITERIA.data())
        )
    );*/
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(from.begin(), to.begin(), weights.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(from.end(), to.end(), weights.end()));
    thrust::sort_by_key(thrust::device, PERSISTENCY_CRITERIA.begin(), PERSISTENCY_CRITERIA.end(), begin, thrust::greater<int>());
    persistency_criteria = PERSISTENCY_CRITERIA;
    

    thrust::device_vector<int> alle_knoten(from.size() + to.size());
    thrust::copy(thrust::device, from.begin(), from.end(), alle_knoten.begin());
    thrust::copy(thrust::device, to.begin(), to.end(), alle_knoten.begin() + from.size());
    
    
    thrust::sort(thrust::device, alle_knoten.begin(), alle_knoten.end());
    
    
    auto new_last = thrust::unique(thrust::device, alle_knoten.begin(), alle_knoten.end());
    
    
    thrust::device_vector<int> unique_knoten(alle_knoten.begin(), new_last);
    
    /*switch to host*/
    thrust::host_vector<int> host_from = from;
    thrust::host_vector<int> host_to = to;
    thrust::host_vector<float> host_weights = weights;
    thrust::host_vector<int> host_persistency_criteria = PERSISTENCY_CRITERIA;
    auto host_first_edge_ = thrust::make_zip_iterator(thrust::make_tuple(host_from.begin(), host_to.begin(), host_weights.begin(), host_persistency_criteria.begin()));
    auto host_after_the_last_edge_ = thrust::make_zip_iterator(thrust::make_tuple(host_from.end(), host_to.end(), host_weights.end(), host_persistency_criteria.end()));
    /*end of switch to host*/

    auto first_edge_ = thrust::make_zip_iterator(thrust::make_tuple(from.begin(), to.begin(), weights.begin(), PERSISTENCY_CRITERIA.begin()));
    auto after_the_last_edge_ = thrust::make_zip_iterator(thrust::make_tuple(from.end(), to.end(), weights.end(), PERSISTENCY_CRITERIA.end()));


    // thrust::device_vector<int> device_node_mapping(unique_knoten.size());
    thrust::host_vector<int> node_mapping(unique_knoten.size());
    unique_knoten_ = unique_knoten;
    for (int i = 0; i < node_mapping.size(); ++i) node_mapping[i] = -1;
    
    compute_node_mapping(host_first_edge_, host_after_the_last_edge_, node_mapping, m);
    /*int category = 0; int ersterSchritt = 1;
    for (int i = 0; i < m; ++i) {
        int u = thrust::get<0>(host_first_edge_[i]);          // int u = thrust::get<0>(first_edge_[i]);
        int v = thrust::get<1>(host_first_edge_[i]);          // int v = thrust::get<1>(first_edge_[i]);
        float theta = thrust::get<2>(host_first_edge_[i]);    // float theta = thrust::get<2>(first_edge_[i]);
        int perCrit = thrust::get<3>(host_first_edge_[i]);    // int perCrit = thrust::get<3>(first_edge_[i]);
        
        
        if (perCrit == 1) {
            
            
            if (ersterSchritt == 1) { node_mapping[u] = category; node_mapping[v] = node_mapping[u]; ersterSchritt = 0; continue; } else {
                if (node_mapping[v] != -1) { node_mapping[u] = node_mapping[v]; continue; }
                if (node_mapping[u] != -1) { node_mapping[v] = node_mapping[u]; continue; }
                if (node_mapping[u] == -1 && node_mapping[v] == -1) { category++; node_mapping[u] = category; node_mapping[v] = node_mapping[u]; }
            }
            
            
        } else {
            
            
            if (node_mapping[u] != -1 && node_mapping[v] == -1) { category++; node_mapping[v] = category; continue; }
            if (node_mapping[v] != -1 && node_mapping[u] == -1) { category++; node_mapping[u] = category; continue; }
            if (node_mapping[u] == -1 && node_mapping[v] == -1) { category++; node_mapping[u] = category; category++; node_mapping[v] = category; }
            
            
        }
    }*/
        
    node_mapping_ = node_mapping;
    thrust::device_vector<int> device_node_mapping = node_mapping;
    
    dCOO contracted_graph = dcoo.contract_cuda(device_node_mapping);
    from_contracted = contracted_graph.get_row_ids();
    to_contracted = contracted_graph.get_col_ids();
    weights_contracted = contracted_graph.get_data();
    

    std::cout << std::endl << "executed" << std::endl;
}

void PersistencyCriteriaSolver::display_from() {
    thrust::for_each(thrust::device, from.begin(), from.end(), print_float_functor());
    std::cout << std::endl;
}

void PersistencyCriteriaSolver::display_to() {
    thrust::for_each(thrust::device, to.begin(), to.end(), print_float_functor());
    std::cout << std::endl;
}

void PersistencyCriteriaSolver::display_weights() {
    thrust::for_each(thrust::device, weights.begin(), weights.end(), print_float_functor());
    std::cout << std::endl;
}

void PersistencyCriteriaSolver::display_input() {
    std::cout << "+++++++++++++++++++++++++++++++++++++++++ INPUT EDGES ++++++++++++++++++++++++++++++++++++++++++\n";
    thrust::for_each(thrust::device, from.begin(), from.end(), print_float_functor()); std::cout << std::endl;
    thrust::for_each(thrust::device, to.begin(), to.end(), print_float_functor()); std::cout << std::endl;
    thrust::for_each(thrust::device, weights.begin(), weights.end(), print_float_functor());
    std::cout << std::endl; std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n";
}

void PersistencyCriteriaSolver::display_abs_weights_per_vertex() {
    std::cout << "\n|||||||||||||||||||||||||||||||||||| abs_weights ||||||||||||||||||||||||||||||||||||\n";
    std::cout << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv             vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n";
    thrust::for_each(thrust::device, all_nodes_.begin(), all_nodes_.end(), print_float_functor()); std::cout << "\n";
    thrust::for_each(thrust::device, abs_weights_.begin(), abs_weights_.end(), print_float_functor()); std::cout << "\n";
    std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ abs_weights ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
    std::cout << "||||||||||||||||||||||||||||||||||||             ||||||||||||||||||||||||||||||||||||\n\n";
}

void PersistencyCriteriaSolver::display_persistency_criteria() {
    std::cout << "++++++++++++++++++++++++++++++++++++ PERSISTENCY CRITERIA FOR THE INPUT DATA ++++++++++++++++++++++++++++++++++++\n";
    thrust::for_each(thrust::device, from.begin(), from.end(), print_float_functor()); std::cout << std::endl;
    thrust::for_each(thrust::device, to.begin(), to.end(), print_float_functor()); std::cout << std::endl;
    thrust::for_each(thrust::device, weights.begin(), weights.end(), print_float_functor()); std::cout << std::endl; std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    thrust::device_vector<char> balken(from.size(), '|'); thrust::device_vector<char> pfeilchen(from.size(), 'v');
    thrust::for_each(thrust::device, balken.begin(), balken.end(), print_char_functor()); std::cout << "\n";
    thrust::for_each(thrust::device, pfeilchen.begin(), pfeilchen.end(), print_char_functor()); std::cout << "\n";
    thrust::for_each(thrust::device, persistency_criteria.begin(), persistency_criteria.end(), print_float_functor());
    std::cout << std::endl << std::endl;
}

void PersistencyCriteriaSolver::display_node_mapping() {
    std::cout << "++++++ node_mapping ++++++\n";
    for (int u : unique_knoten_) std::cout << u << "\t"; std::cout << std::endl;
    for (int m : node_mapping_) std::cout << m << "\t"; std::cout << std::endl;
}

void PersistencyCriteriaSolver::display_contracted_graph() {
    std::cout << "\n++++++ contracted graph (minor of G) ++++++\n";
    for (int v: from_contracted) std::cout << v << "\t"; std::cout << "\n";
    for (int b: to_contracted) std::cout << b << "\t"; std::cout << "\n";
    for (float g: weights_contracted) std::cout << g << "\t"; std::cout << "\n";
}

void PersistencyCriteriaSolver::compare() {
    std::cout << "\n++++++++ Comparison of |E| before and after contraction ++++++++\n";
    std::cout << "Initial |V|:\t" << unique_knoten_.size() << std::endl;
    std::cout << "Initial |E|:\t" << from.size() << std::endl;
    /*thrust::device_vector<int> copy_node_mapping(node_mapping_.size());
    thrust::copy(thrust::device, node_mapping_.begin(), node_mapping_.begin(), copy_node_mapping.begin());
    thrust::sort(thrust::device, copy_node_mapping.begin(), copy_node_mapping.end());
    auto new_end = thrust::unique(thrust::device, copy_node_mapping.begin(), copy_node_mapping.end());
    thrust::device_vector<int> copy_node_mapping_(copy_node_mapping.begin(), new_end);
    std::cout << "|V| after contraction (including loops):\t" << copy_node_mapping_.size() << std::endl;*/
    std::cout << "|E| after contraction (including loops):\t" << from_contracted.size() << std::endl;
    int amount_of_contracted_edges = thrust::count(thrust::device, persistency_criteria.begin(), persistency_criteria.end(), 1.0);
    float contraction_rate = (float) amount_of_contracted_edges / persistency_criteria.size();
    contraction_rate *= 100.0;
    std::cout.precision(2);
    std::cout << contraction_rate << "% of edges were contracted.\n";
}

void PersistencyCriteriaSolver::compute_per_crit(auto first, auto last, thrust::device_vector<float> &abs_weights, thrust::device_vector<int> &PERSISTENCY_CRITERIA) {
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    /*auto first_edge = thrust::make_zip_iterator(thrust::make_tuple(from.begin(), to.begin(), weights.begin(), counter.begin()));
    auto after_the_last_edge = thrust::make_zip_iterator(thrust::make_tuple(from.end(), to.end(), weights.end(), counter.end()));*/
    thrust::for_each(
        thrust::device,
        first,
        last,
        __PER__CRIT__FUNCTOR__<float, int>(
            thrust::raw_pointer_cast(abs_weights.data()), thrust::raw_pointer_cast(PERSISTENCY_CRITERIA.data())
        )
    );
}

void PersistencyCriteriaSolver::compute_node_mapping(auto first, auto last, thrust::host_vector<int> &node__mapping__, auto m) {
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    int category = 0; int ersterSchritt = 1;
    for (int i = 0; i < m; ++i) {
        int u = thrust::get<0>(first[i]);          // int u = thrust::get<0>(first_edge_[i]);
        int v = thrust::get<1>(first[i]);          // int v = thrust::get<1>(first_edge_[i]);
        float theta = thrust::get<2>(first[i]);    // float theta = thrust::get<2>(first_edge_[i]);
        int perCrit = thrust::get<3>(first[i]);    // int perCrit = thrust::get<3>(first_edge_[i]);
        if (perCrit == 1) {
            if (ersterSchritt == 1) { node__mapping__[u] = category; node__mapping__[v] = node__mapping__[u]; ersterSchritt = 0; continue; } else {
                if (node__mapping__[v] != -1) { node__mapping__[u] = node__mapping__[v]; continue; }
                if (node__mapping__[u] != -1) { node__mapping__[v] = node__mapping__[u]; continue; }
                if (node__mapping__[u] == -1 && node__mapping__[v] == -1) { category++; node__mapping__[u] = category; node__mapping__[v] = node__mapping__[u]; }
            }
        } else {
            if (node__mapping__[u] != -1 && node__mapping__[v] == -1) { category++; node__mapping__[v] = category; continue; }
            if (node__mapping__[v] != -1 && node__mapping__[u] == -1) { category++; node__mapping__[u] = category; continue; }
            if (node__mapping__[u] == -1 && node__mapping__[v] == -1) { category++; node__mapping__[u] = category; category++; node__mapping__[v] = category; }
        }
    }
}