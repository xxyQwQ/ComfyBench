- `ImageBatchSplitter __Inspire`: The ImageBatchSplitter node is designed to split a batch of images into smaller batches or individual images, based on a specified split count. It can also pad the output with empty images if the requested split count exceeds the number of available images, ensuring the output always matches the requested size.
    - Inputs:
        - `images` (Required): The collection of images to be split. This parameter is crucial for determining the subset of images to be processed and split according to the specified count. Type should be `IMAGE`.
        - `split_count` (Required): Specifies the number of splits or individual images to be extracted from the input batch. This count directly influences the size and composition of the output batches. Type should be `INT`.
    - Outputs:
        - `image`: A tuple of tensors, each representing a split batch or an individual image from the original input batch, potentially including padding with empty images to meet the requested split count. Type should be `IMAGE`.