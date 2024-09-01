- `VHS_VAEDecodeBatched`: This node is designed for batch processing of latent representations to decode them back into images using a specified VAE model. It efficiently handles large sets of data by processing them in smaller, manageable batches.
    - Parameters:
        - `per_batch`: Specifies the number of samples to be processed in each batch. This allows for efficient memory management and processing speed optimization. Type should be `INT`.
    - Inputs:
        - `samples`: The latent representations to be decoded into images. It's crucial for reconstructing the original or modified images from their compressed form. Type should be `LATENT`.
        - `vae`: The VAE model used for decoding the latent representations. It defines the architecture and parameters for the decoding process. Type should be `VAE`.
    - Outputs:
        - `image`: The decoded images, reconstructed from the provided latent representations. Type should be `IMAGE`.