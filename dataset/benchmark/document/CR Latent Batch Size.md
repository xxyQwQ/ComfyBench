- `CR Latent Batch Size`: This node is designed to adjust the batch size of latent representations by replicating the input samples to match the specified batch size. It facilitates the manipulation of data batch sizes for downstream processing or model inference.
    - Inputs:
        - `latent` (Required): The latent representation of data, typically a tensor, that is to be adjusted in terms of batch size. This input is crucial for determining the structure and content of the data to be replicated. Type should be `LATENT`.
        - `batch_size` (Required): An integer specifying the desired batch size. This parameter dictates how many times the input samples are replicated to achieve the specified batch size. Type should be `INT`.
    - Outputs:
        - `latent`: The output latent representation with the adjusted batch size, achieved by replicating the input samples as specified. Type should be `LATENT`.
