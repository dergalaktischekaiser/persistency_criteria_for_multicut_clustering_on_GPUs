/*source: https://github.com/pawelswoboda/RAMA */
#include "dCOO.h"
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include "time_measure_util.h"
#include "rama_utils.h"


__global__ void map_nodes(const int num_edges, const int* const __restrict__ node_mapping, int* __restrict__ rows, int* __restrict__ cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int e = tid; e < num_edges; e += num_threads)
    {
        assert(rows[e] < cols[e]);
        const int i = node_mapping[rows[e]];
        const int j = node_mapping[cols[e]];
        rows[e] = min(i, j);
        cols[e] = max(i, j);
    }
}

void dCOO::init(const bool is_sorted)
{
    assert(col_ids.size() == data.size());
    assert(row_ids.size() == data.size());
    std::cout << "\nassertion 1 passed\n";
    if(is_sorted)
    {
        assert(thrust::is_sorted(row_ids.begin(), row_ids.end())); 
        assert(thrust::is_sorted(thrust::make_zip_iterator(thrust::make_tuple(row_ids.begin(), col_ids.begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(row_ids.end(), col_ids.end())))); 
    }
    else
    {
        if (is_directed_)
            sort_edge_nodes(row_ids, col_ids);
        coo_sorting(row_ids, col_ids, data);
        // now row indices are non-decreasing
        assert(thrust::is_sorted(row_ids.begin(), row_ids.end()));
        std::cout << "assertion 2 passed\n";
    } 

    if(cols_ == 0)
        cols_ = *thrust::max_element(col_ids.begin(), col_ids.end()) + 1;
    assert(cols_ > *thrust::max_element(col_ids.begin(), col_ids.end()));
    if(rows_ == 0)
        rows_ = row_ids.back() + 1;
    assert(rows_ > *thrust::max_element(row_ids.begin(), row_ids.end()));
    if (!is_directed_)
        assert(rows_ == cols_);
    std::cout << "all assertions passed\n";
}

dCOO dCOO::contract_cuda(const thrust::device_vector<int>& node_mapping)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    assert(is_directed_);

    const int numThreads = 256;

    thrust::device_vector<int> new_row_ids = row_ids;
    thrust::device_vector<int> new_col_ids = col_ids;
    thrust::device_vector<float> new_data = data;

    int num_edges = new_row_ids.size();
    int numBlocks = ceil(num_edges / (float) numThreads);
    map_nodes<<<numBlocks, numThreads>>>(num_edges, 
            thrust::raw_pointer_cast(node_mapping.data()), 
            thrust::raw_pointer_cast(new_row_ids.data()), 
            thrust::raw_pointer_cast(new_col_ids.data()));

    coo_sorting(new_row_ids, new_col_ids, new_data); // in-place sorting by rows.

    auto first = thrust::make_zip_iterator(thrust::make_tuple(new_row_ids.begin(), new_col_ids.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(new_row_ids.end(), new_col_ids.end()));

    thrust::device_vector<int> out_rows(num_edges);
    thrust::device_vector<int> out_cols(num_edges);
    auto first_output = thrust::make_zip_iterator(thrust::make_tuple(out_rows.begin(), out_cols.begin()));
    thrust::device_vector<float> out_data(num_edges);

    auto new_end = thrust::reduce_by_key(first, last, new_data.begin(), first_output, out_data.begin());
    int new_num_edges = std::distance(out_data.begin(), new_end.second);
    out_rows.resize(new_num_edges);
    out_cols.resize(new_num_edges);
    out_data.resize(new_num_edges);

    int out_num_rows = out_rows.back() + 1;
    int out_num_cols = *thrust::max_element(out_cols.begin(), out_cols.end()) + 1;

    return dCOO(out_num_rows, out_num_cols, std::move(out_cols), std::move(out_rows), std::move(out_data), is_directed_, true);
}

struct is_diagonal
{
    __host__ __device__
        bool operator()(thrust::tuple<int,int,float> t)
        {
            return thrust::get<0>(t) == thrust::get<1>(t);
        }
};

void dCOO::remove_diagonal()
{
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));

    auto new_last = thrust::remove_if(begin, end, is_diagonal());
    int new_num_edges = std::distance(begin, new_last);
    col_ids.resize(new_num_edges);
    row_ids.resize(new_num_edges);
    data.resize(new_num_edges);
}

struct diag_func
{
    float* d;
    __host__ __device__
        void operator()(thrust::tuple<int,int,float> t)
        {
            if(thrust::get<0>(t) == thrust::get<1>(t))
            {
                assert(d[thrust::get<0>(t)] == 0.0);
                d[thrust::get<0>(t)] = thrust::get<2>(t);
            }
        }
};

thrust::device_vector<float> dCOO::diagonal() const
{
    thrust::device_vector<float> d(std::max(rows(), cols()), 0.0);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));

    thrust::for_each(begin, end, diag_func({thrust::raw_pointer_cast(d.data())})); 

    return d;
}

thrust::device_vector<int> dCOO::compute_row_offsets() const
{
    return compute_offsets(row_ids, rows_ - 1);
}

float dCOO::sum() const
{
    return thrust::reduce(data.begin(), data.end(), (float) 0.0, thrust::plus<float>());
}

float dCOO::min() const
{
    return *thrust::min_element(data.begin(), data.end()); 
}

float dCOO::max() const
{
    return *thrust::max_element(data.begin(), data.end()); 
}

dCOO dCOO::export_undirected() const
{
    assert(is_directed_);
    thrust::device_vector<int> row_ids_u, col_ids_u;
    thrust::device_vector<float> data_u;

    std::tie(row_ids_u, col_ids_u, data_u) = to_undirected(row_ids, col_ids, data);
    return dCOO(std::move(col_ids_u), std::move(row_ids_u), std::move(data_u), false);
}

dCOO dCOO::export_directed() const
{
    assert(!is_directed_);
    thrust::device_vector<int> row_ids_d, col_ids_d;
    thrust::device_vector<float> data_d;

    std::tie(row_ids_d, col_ids_d, data_d) = to_directed(row_ids, col_ids, data);
    return dCOO(std::move(col_ids_d), std::move(row_ids_d), std::move(data_d), true);
}

struct is_in_range
{
    const float lb;
    const float ub;

      __host__ __device__
            bool operator()(const float x)
            {
                if(x >= lb && x <= ub)
                    return true;
                else
                    return false;
            }

      __host__ __device__
            bool operator()(const thrust::tuple<int,int,float> t)
            {
                return operator()(thrust::get<2>(t));
            }
};

dCOO dCOO::export_filtered(const float lb, const float ub) const
{
    assert(lb <= ub);
    const int new_nnz = thrust::count_if(data.begin(), data.end(), is_in_range({lb,ub}));
    thrust::device_vector<int> col_ids_f(new_nnz), row_ids_f(new_nnz);
    thrust::device_vector<float> data_f(new_nnz);

    auto first = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));

    auto first_f = thrust::make_zip_iterator(thrust::make_tuple(col_ids_f.begin(), row_ids_f.begin(), data_f.begin()));

    thrust::copy_if(first, last, first_f, is_in_range({lb,ub}));

    return dCOO(rows(), cols(), 
            std::move(col_ids_f), std::move(row_ids_f), std::move(data_f), is_directed_, true); 
}

void dCOO::print() const
{
    std::cout<<"A: \n";
    print_vector(row_ids, "row_ids");
    print_vector(col_ids, "col_ids");
    print_vector(data   , "data   ");
    std::cout<<"\n";
}