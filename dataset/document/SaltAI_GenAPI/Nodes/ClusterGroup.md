# ClusterGroup
## Documentation
- Class name: `ClusterGroup`
- Category: `SALT/CLUSTERING`
- Output node: `False`

The ClusterGroup node is designed for clustering a set of features into groups based on their similarity. It utilizes dimensionality reduction techniques followed by a clustering algorithm to categorize the data into clusters, and generates visualizations to represent the clustering and distribution of data points.
## Input types
### Required
- **`features`**
    - A collection of features to be clustered. These features are the basis for the dimensionality reduction and subsequent clustering process.
    - Comfy dtype: `NP_ARRAY`
    - Python dtype: `numpy.ndarray or similar array-like structure`
- **`n_components`**
    - The number of dimensions to reduce the feature space to before clustering. This parameter influences the granularity of the clustering process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`min_cluster_size`**
    - The minimum size of clusters to be formed. This parameter helps in controlling the sensitivity of the clustering algorithm to small clusters.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`max_cluster_size`**
    - The maximum size of clusters to be allowed. This parameter sets an upper limit on cluster sizes, preventing overly large clusters.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`labels`**
    - Comfy dtype: `LIST`
    - The labels assigned to each data point indicating the cluster it belongs to.
    - Python dtype: `numpy.ndarray`
- **`clustering_img`**
    - Comfy dtype: `IMAGE`
    - A visualization of the clustered data points, with different colors representing different clusters.
    - Python dtype: `object`
- **`distribution_img`**
    - Comfy dtype: `IMAGE`
    - A bar plot showing the distribution of data points across the top-level categories or clusters.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ClusterGroup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "features": ("NP_ARRAY", ),
                "n_components": ("INT", {"default": 4, "min": 1}),
                "min_cluster_size": ("INT", {"default": 400, "min": 1}),
                "max_cluster_size": ("INT", {"default": 30000, "min": 1}),
            }
        }

    RETURN_TYPES = ("LIST", "IMAGE", "IMAGE")
    RETURN_NAMES = ("labels", "clustering_img", "distribution_img")

    FUNCTION = "claster"
    CATEGORY = f"{MENU_NAME}/CLUSTERING"

    def claster(self, features, n_components, min_cluster_size, max_cluster_size):
        try:
            umap_reducer = umap.UMAP(random_state=42, n_components=n_components)
            X_umap = umap_reducer.fit_transform(features)
            projection = TSNE().fit_transform(X_umap)
            
            # Apply HDBSCAN for clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                max_cluster_size=max_cluster_size,
            )
            labels = clusterer.fit_predict(X_umap)
            
            # Visualize the clustering results with different colors for each category
            plt.figure(figsize=(12, 8))
            palette = sns.color_palette("hsv", len(set(labels)))
            sns.scatterplot(x=projection[:, 0], y=projection[:, 1], hue=labels, legend='full', palette=palette)
            plt.title('UMAP Clustering with HDBSCAN')
            plt.legend(title='Top Category')
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            clustering_img = pil2tensor(Image.open(buf))
            
            # Count the number of items in each top-level category
            category_counts = pd.Series(labels).value_counts().reset_index()
            category_counts.columns = ['top_category', 'count']
            
            # Plot the bar plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='top_category', y='count', data=category_counts, palette='viridis')
            plt.title('Number of Items in Each Top-Level Category')
            plt.xlabel('Top-Level Category')
            plt.ylabel('Number of Items')
            plt.xticks(rotation=90)
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            distribution_img = pil2tensor(Image.open(buf))
        except:
            black_img = Image.new("RGB", (100, 100), (0, 0, 0))
            return ([], pil2tensor(black_img), pil2tensor(black_img))
        return (labels, clustering_img, distribution_img)

```
