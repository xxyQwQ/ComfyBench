- `StableZero123_Conditioning_Batched`: This node is designed to process conditioning data in batches for the StableZero123 model, optimizing the conditioning process for efficiency and scalability. It focuses on handling multiple conditioning inputs simultaneously, applying model-specific adjustments to prepare them for the StableZero123 model's requirements.
    - Parameters:
        - `width`: The desired width of the output image, influencing the dimensionality of the generated image. Type should be `INT`.
        - `height`: The desired height of the output image, influencing the dimensionality of the generated image. Type should be `INT`.
        - `batch_size`: The number of images to process in a single batch, affecting the efficiency and speed of the conditioning process. Type should be `INT`.
        - `elevation`: The elevation angle for 3D model viewing, affecting the perspective from which the model is rendered. Type should be `FLOAT`.
        - `azimuth`: The azimuth angle for 3D model viewing, affecting the orientation of the model in the rendered image. Type should be `FLOAT`.
        - `elevation_batch_increment`: The incremental change in elevation angle across the batch, allowing for varied perspectives in a single batch. Type should be `FLOAT`.
        - `azimuth_batch_increment`: The incremental change in azimuth angle across the batch, allowing for varied orientations in a single batch. Type should be `FLOAT`.
    - Inputs:
        - `clip_vision`: Specifies the CLIP vision model to be used for conditioning, affecting how input images are interpreted and processed. Type should be `CLIP_VISION`.
        - `init_image`: The initial image to start the generation process, serving as a base for further modifications. Type should be `IMAGE`.
        - `vae`: The variational autoencoder used for encoding and decoding images, integral to the image transformation process. Type should be `VAE`.
    - Outputs:
        - `positive`: The positive conditioning output, tailored for promoting certain features or aspects in the generated image. Type should be `CONDITIONING`.
        - `negative`: The negative conditioning output, tailored for suppressing certain features or aspects in the generated image. Type should be `CONDITIONING`.
        - `latent`: The latent representation of the image, used for further processing or generation steps. Type should be `LATENT`.