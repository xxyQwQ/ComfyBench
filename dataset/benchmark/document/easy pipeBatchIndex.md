- `easy pipeBatchIndex`: The `easy pipeBatchIndex` node is designed to extract a specific batch of samples from a pipeline's sample collection, based on a given batch index and length. This functionality is crucial for processing or analyzing subsets of data within larger datasets, enabling targeted operations on specific segments of the sample collection.
    - Inputs:
        - `pipe` (Required): Specifies the pipeline from which a batch of samples is to be extracted. It is essential for identifying the source collection of samples. Type should be `PIPE_LINE`.
        - `batch_index` (Required): Determines the starting index of the batch to be extracted from the pipeline. It allows for precise selection of data segments. Type should be `INT`.
        - `length` (Required): Defines the number of samples to be extracted from the specified batch index. This parameter sets the size of the batch to be processed. Type should be `INT`.
    - Outputs:
        - `pipe`: Returns a modified pipeline containing only the samples from the specified batch index and length. This enables focused analysis or processing of a particular data subset. Type should be `PIPE_LINE`.