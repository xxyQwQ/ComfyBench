- `ADE_UpscaleAndVAEEncode`: The ADE_UpscaleAndVAEEncode node is designed for processing images by first upscaling them to a higher resolution and then encoding them into a latent representation using a Variational Autoencoder (VAE). This node is part of the AnimateDiff suite, specifically tailored for enhancing image quality before applying further generative or transformational processes.
    - Parameters:
        - `scale_method`: The 'scale_method' parameter defines the method used for upscaling the image. It influences the quality of the upscaled image. Type should be `COMBO[STRING]`.
        - `crop`: The 'crop' parameter specifies the cropping method applied after upscaling, affecting the final image composition. Type should be `COMBO[STRING]`.
    - Inputs:
        - `image`: The 'image' parameter represents the input image to be upscaled and encoded. It plays a crucial role in determining the quality and resolution of the final latent representation. Type should be `IMAGE`.
        - `vae`: The 'vae' parameter specifies the Variational Autoencoder model used for encoding the upscaled image into its latent representation. It affects the encoding efficiency and the quality of the generated latent space. Type should be `VAE`.
        - `latent_size`: The 'latent_size' parameter indicates the size of the latent representation to be generated. It determines the dimensions of the output latent space. Type should be `LATENT`.
    - Outputs:
        - `latent`: The output is a latent representation of the input image, encoded by the VAE after upscaling. It captures the essential features of the image in a compressed form, suitable for further generative tasks. Type should be `LATENT`.