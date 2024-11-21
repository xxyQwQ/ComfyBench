- `ClusterGroup`: The ClusterGroup node is designed for clustering a set of features into groups based on their similarity. It utilizes dimensionality reduction techniques followed by a clustering algorithm to categorize the data into clusters, and generates visualizations to represent the clustering and distribution of data points.
    - Inputs:
        - `features` (Required): A collection of features to be clustered. These features are the basis for the dimensionality reduction and subsequent clustering process. Type should be `NP_ARRAY`.
        - `n_components` (Required): The number of dimensions to reduce the feature space to before clustering. This parameter influences the granularity of the clustering process. Type should be `INT`.
        - `min_cluster_size` (Required): The minimum size of clusters to be formed. This parameter helps in controlling the sensitivity of the clustering algorithm to small clusters. Type should be `INT`.
        - `max_cluster_size` (Required): The maximum size of clusters to be allowed. This parameter sets an upper limit on cluster sizes, preventing overly large clusters. Type should be `INT`.
    - Outputs:
        - `labels`: The labels assigned to each data point indicating the cluster it belongs to. Type should be `LIST`.
        - `clustering_img`: A visualization of the clustered data points, with different colors representing different clusters. Type should be `IMAGE`.
        - `distribution_img`: A bar plot showing the distribution of data points across the top-level categories or clusters. Type should be `IMAGE`.