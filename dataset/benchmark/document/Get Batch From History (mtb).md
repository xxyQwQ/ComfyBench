- `Get Batch From History (mtb)`: The MTB_GetBatchFromHistory node is designed to retrieve a batch of data from a historical dataset based on specified parameters. It enables conditional fetching of data, allowing for dynamic data loading and manipulation within a pipeline.
    - Inputs:
        - `enable` (Required): Determines whether the node should attempt to load data from history. If disabled, it can pass through an alternative image if provided. Type should be `BOOLEAN`.
        - `count` (Required): Specifies the number of historical data entries to retrieve. A count of 0 disables data loading. Type should be `INT`.
        - `offset` (Required): Defines the starting point within the historical dataset from which to begin data retrieval, allowing for pagination or skipping entries. Type should be `INT`.
        - `internal_count` (Required): A hacky parameter used to invalidate the node's cache, forcing a reload under certain conditions. Type should be `INT`.
        - `passthrough_image` (Optional): An optional image to pass through when data loading is disabled or the count is 0, providing a fallback mechanism. Type should be `IMAGE`.
    - Outputs:
        - `images`: The batch of images retrieved from history, or a zero tensor if no data is found or loading is disabled. Type should be `IMAGE`.
