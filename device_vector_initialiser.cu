#include "device_vector_initialiser.h"

#include <tuple>

thrust::device_vector<int> device_int_vector_initializr (
    const int n,
    const INPUT_DATA_TUPLE &_input_data
) {
    try {
        if (n <= 1) std::cout << "";
        else throw (n);
    } catch (int _N) {
        std::cout << "Forbidden memory access! Index " << _N << " lies outside of the allocated memory!\n";
        thrust::device_vector<int> i_am_empty;
        return i_am_empty;
    }
    if (n == 0) {
        std::vector<int> from_or_to_cpu_ = std::get<0>(_input_data);
        thrust::device_vector<int> from_or_to_gpu_(from_or_to_cpu_.size());
        for (int i = 0; i < from_or_to_cpu_.size(); ++i) from_or_to_gpu_[i] = from_or_to_cpu_[i];
        return from_or_to_gpu_;
    }
    std::vector<int> from_or_to_cpu_ = std::get<1>(_input_data);
    thrust::device_vector<int> from_or_to_gpu_(from_or_to_cpu_.size());
    for (int i = 0; i < from_or_to_cpu_.size(); ++i) from_or_to_gpu_[i] = from_or_to_cpu_[i];
    return from_or_to_gpu_;
}

thrust::device_vector<float> device_float_vector_initializr (
    const int n,
    const INPUT_DATA_TUPLE &_input_data
) {
    try {
        if (n == 2) std::cout << "";
        else throw (n);
    } catch (int _N) {
        std::cout << "Don't forget to extract the <float> vector in the initialization from INPUT_DATA_TUPLE!\n";
        thrust::device_vector<float> i_am_empty;
        return i_am_empty;
    }
    std::vector<float> from_or_to_cpu_ = std::get<2>(_input_data);
    thrust::device_vector<float> from_or_to_gpu_(from_or_to_cpu_.size());
    for (int i = 0; i < from_or_to_cpu_.size(); ++i) from_or_to_gpu_[i] = from_or_to_cpu_[i];
    return from_or_to_gpu_;
}

bool validate_input(const thrust::device_vector<int> &from, const thrust::device_vector<int> &to, const thrust::device_vector<float> &weights) {
    try {
        if (from.size() == to.size() && to.size() == weights.size()) std::cout << "Validated.\n";
        else throw (from);
    } catch (int _N) {
        std::cout << "Invalid input!\n";
        thrust::device_vector<int> i_am_empty;
        return false;
    }
    return true;
}