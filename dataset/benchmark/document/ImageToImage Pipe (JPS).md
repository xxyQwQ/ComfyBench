- `ImageToImage Pipe (JPS)`: The ImageToImage Pipe node is designed to process image-to-image settings, extracting and returning various parameters related to image transformation processes. It abstracts the complexity of configuring settings for image-to-image operations, providing a streamlined interface for specifying and retrieving transformation parameters.
    - Inputs:
        - `img2img_settings` (Required): Specifies the settings for image-to-image transformation, including strengths and configurations for inpainting, unsampling, and other related processes. It is crucial for defining how the image will be processed and transformed. Type should be `BASIC_PIPE`.
    - Outputs:
        - `img2img_strength`: The strength of the image-to-image transformation. Type should be `FLOAT`.
        - `inpaint_strength`: The strength of the inpainting process in the transformation. Type should be `FLOAT`.
        - `inpaint_grow_mask`: Specifies how much the inpaint mask should grow. Type should be `INT`.
        - `unsampler_strength`: The strength of the unsampling process. Type should be `FLOAT`.
        - `unsampler_cfg`: Configuration for the unsampler process. Type should be `FLOAT`.
        - `unsampler_sampler`: The sampler used in the unsampling process. Type should be `COMBO[STRING]`.
        - `unsampler_scheduler`: The scheduler used for unsampling. Type should be `COMBO[STRING]`.
