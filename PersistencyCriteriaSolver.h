#pragma once

#include <string>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct print_float_functor {
    __host__ __device__
    void operator() (float _x) {
        printf("%f\t", _x);
    }
};

struct abs_functor {
    __device__
    float operator() (float _x) {
        return std::abs(_x);
    }
};

struct print_char_functor {
    __host__ __device__
    void operator() (char _x) {
        printf("%c\t\t", _x);
    }
};


template <typename T, typename S>
struct __PER__CRIT__FUNCTOR__ {
    T *abs_weights_; S *persistency_criteria_;
    __PER__CRIT__FUNCTOR__(T *_abs_weights, S *_persistency_criteria) : abs_weights_(_abs_weights), persistency_criteria_(_persistency_criteria) {};
    __device__
    void operator() (thrust::tuple<int, int, float, int> kante) {
        int u = thrust::get<0>(kante);
        int v = thrust::get<1>(kante);
        float theta = thrust::get<2>(kante);
        int kantennummer = thrust::get<3>(kante);
        float sum_ohne_theta_f = abs_weights_[u] - std::abs(theta);
        if (theta >= sum_ohne_theta_f) persistency_criteria_[kantennummer] = 1;
        sum_ohne_theta_f = abs_weights_[v] - std::abs(theta);
        if (theta >= sum_ohne_theta_f) persistency_criteria_[kantennummer] = 1;
    }
};

class PersistencyCriteriaSolver {
    private:
        thrust::device_vector<int> from;
        thrust::device_vector<int> to;
        thrust::device_vector<float> weights;
        thrust::device_vector<int> all_nodes_;
        thrust::device_vector<float> abs_weights_;
        thrust::device_vector<int> persistency_criteria;
        thrust::device_vector<int> unique_knoten_;
        thrust::device_vector<int> node_mapping_;
        thrust::device_vector<int> from_contracted;
        thrust::device_vector<int> to_contracted;
        thrust::device_vector<float> weights_contracted;
        // auto time_for_persistency_criteria;
    public:
        PersistencyCriteriaSolver() {}
        PersistencyCriteriaSolver(
            const thrust::device_vector<int> &_from,
            const thrust::device_vector<int> &_to,
            const thrust::device_vector<float> &_weights
        );
        void execute();
        void display_from();
        void display_to();
        void display_weights();
        void display_input();
        void display_abs_weights_per_vertex();
        void display_persistency_criteria();
        void display_node_mapping();
        void display_contracted_graph();
        void compare();

        void compute_per_crit(auto first, auto last, thrust::device_vector<float> &abs_weights_, thrust::device_vector<int> &per_crit_);
        void compute_node_mapping(auto first, auto last, thrust::host_vector<int> &node_mapping_, auto m);
};