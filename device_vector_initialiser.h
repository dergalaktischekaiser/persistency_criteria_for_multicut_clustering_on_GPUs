#pragma once

#include <iostream>
#include <tuple>
#include <vector>
#include <thrust/device_vector.h>

typedef std::tuple<std::vector<int>, std::vector<int>, std::vector<float>> INPUT_DATA_TUPLE;

thrust::device_vector<int> device_int_vector_initializr(
    const int n,
    const INPUT_DATA_TUPLE &_input_data
);

thrust::device_vector<float> device_float_vector_initializr (
    const int n,
    const INPUT_DATA_TUPLE &_input_data
);

bool validate_input(const thrust::device_vector<int> &from, const thrust::device_vector<int> &to, const thrust::device_vector<float> &weights);