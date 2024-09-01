- `VAEEncodeTiled`: The VAEEncodeTiled node is designed for encoding images into a latent space representation, specifically handling images in a tiled manner. This approach allows for processing larger images by dividing them into smaller, manageable tiles, encoding each separately, and then combining the results.
    - Parameters:
        - `tile_size`: The 'tile_size' parameter determines the dimensions of the tiles into which the image is divided for encoding. It affects the granularity of the tiling process and can impact the encoding performance and quality. Type should be `INT`.
    - Inputs:
        - `pixels`: The 'pixels' parameter represents the image data to be encoded. It is crucial for defining the visual content that will be transformed into a latent representation. Type should be `IMAGE`.
        - `vae`: The 'vae' parameter specifies the Variational Autoencoder model used for the encoding process. It plays a key role in determining how the image data is transformed into the latent space. Type should be `VAE`.
    - Outputs:
        - `latent`: The output is a latent space representation of the input image, encoded in a tiled manner. It captures the essential features of the image in a compressed form. Type should be `LATENT`.