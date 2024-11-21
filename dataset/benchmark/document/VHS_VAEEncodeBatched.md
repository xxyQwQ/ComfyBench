- `VHS_VAEEncodeBatched`: This node is designed for batch processing of images through a Variational Autoencoder (VAE) to encode them into a latent space representation. It efficiently handles large sets of images by dividing them into smaller batches, encoding each batch separately, and then aggregating the results. This approach optimizes resource utilization and accelerates the encoding process, making it suitable for applications requiring the transformation of images into their latent representations for further processing or analysis.
    - Inputs:
        - `pixels` (Required): The 'pixels' parameter represents the images to be encoded into latent space. It is crucial for defining the input data that will undergo the encoding process, directly influencing the node's output by determining the characteristics of the generated latent representations. Type should be `IMAGE`.
        - `vae` (Required): The 'vae' parameter specifies the Variational Autoencoder model to be used for encoding the images. It plays a pivotal role in the transformation process, as the model's architecture and trained parameters directly affect the quality and characteristics of the encoded latent space. Type should be `VAE`.
        - `per_batch` (Required): The 'per_batch' parameter determines the number of images to be processed in each batch. It allows for flexible control over the batch size, balancing between computational efficiency and resource consumption. Type should be `INT`.
    - Outputs:
        - `latent`: The output is a latent space representation of the input images, encoded by the specified VAE model. This representation is crucial for downstream tasks that require a compressed yet informative version of the original data. Type should be `LATENT`.