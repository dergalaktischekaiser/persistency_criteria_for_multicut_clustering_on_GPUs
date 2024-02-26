# Persistency criteria for Multicut-Clustering on GPUs

We propose a GPU-based approach for persistency criteria computation for multicut clustering. Our approach introduces a solver which determines the persistency criteria for any common weighted graph. The solver is implemented in CUDA with ``thrust`` template library. The persistency criteria were previously defined in [this paper](https://arxiv.org/pdf/1812.01426.pdf).
The solver requires a .txt input file. The input file should be organised as follows:
```
MULTICUT
from_1 to_1 weight_1
from_2 to_2 weight_2
...    ...  ...
from_m to_m weight_m
```

## How does the solver validate the persistency criteria?
Given a graph $G(V, E, \Theta)$, the solver investigvtes, whether for each edge $f\in E$ the following condition
$$\theta_f \geq \sum\limits_{e\in\delta(U)\setminus\{f\}}| \theta_e|,$$
where $\theta_f\in\mathbb{R}$ is the weight of $f$, applies.

If the condition holds, then the solver writes ``1`` to the corresponding index of $f$ in ``thrust::device_vector<int> persistency_criteria`` which is initially filled with zeros. Otherwise, the value ``0`` remains in the vector.

Beyond the computation of the persistency criteria, the solver is capable of obtaining a node mapping in order to perform edge contraction later. Node mapping matches every vertex of $G$ to a natural number in order to establish a multicut-based clustering of the graph.

To run the solver, you need to run ``mkdir build`` to create the ``build`` directory. Navigate to the directory with ``cd build``. Create ``multicut_instance`` directory. Insert your input .txt-file into the directory. Fill the file with data (your input graph). Make sure your data meets the input requirements descirbed above. Run ``cmake ..`` and ``cmake --build .`` to build the project. Then run ``./PersistencyCriteriaForMulticutClusteringOnGPUs <your file_name>.txt``.
