# Dataset
We recommend that users to adopt the standard h5 format data (also known as HDF5), which is common among data preprocessing and feature selection. Data in other formats (such as .csv, .txt) can be converted into h5 format data for input.

# Pre-processing
In most cases, the scRNA-seq datasets always need different data preprocessing steps, such as cell gene filtering, logarithmization, highly-variable genes selection. These are aggregated into an internal function named `normalize`. It is worth noting that the rule for selecting the number of highly variable genes is not less than 50 since less than 50 can cause errors in building the cell-PPMI graph. The reason being that in order to calculate the KNN graph faster when building the cell graph, we performed the PCA operation on the data and reduced the dimension to 50 dimensions. Therefore, if the data itself is less than 50 dimensions, the code will report the $ValueError$. However, in reality, the use cases with less than 50 genes are pretty rare in practice. One of the workarounds is to modify the PCA source code to be less than 50 dimensions.

# Cell graph
The cell-PPMI graph is obtained by calculating the co-occurrence probability of cells through PPMI which is a symmetric undirected graph. Therefore, if the user wants to use their own cell graph for input, we recommend a symmetric undirected graph as the cell-PPMI graph. Furthermore, it is necessary to ensure that the parameter $k$ in the function `get_adj` is less than or equal to the number of cells in the dataset (`n_neighbors` > `n_samples`); otherwise, it will report the $ValueError$.

# Batch settings
scMGCA performs batch learning for the datasets with more than 25,000 cells. Therefore, when using scMGCA to cluster the dataset with more than 25,000 cells, the user should select the function `get_adj_batch` in $graph_function.py$ to construct the cell graph and the class `SCMGCAL` in $scmgca.py$ as the model for calculation; otherwise, an out of memory (OOM) error can occur if memory resources are not enough.
