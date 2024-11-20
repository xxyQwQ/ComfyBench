---
tags:
- Sampling
---

# Ultimate SD Upscale (Custom Sample)
## Documentation
- Class name: `UltimateSDUpscaleCustomSample`
- Category: `image/upscaling`
- Output node: `False`

This node specializes in enhancing image quality through upscaling, leveraging advanced techniques to refine image details and textures. It extends the capabilities of standard upscaling by allowing for custom model and sampling parameters, providing a tailored approach to image enhancement.
## Input types
### Required
- **`image`**
    - The original image to be upscaled. This is the starting point for the upscaling process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`model`**
    - The generative model used for enhancing the image, which plays a crucial role in the upscaling process.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`positive`**
    - Positive conditioning text to guide the model in enhancing specific aspects of the image.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Negative conditioning text to instruct the model on what aspects to avoid or minimize in the image.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`vae`**
    - The variational autoencoder used in conjunction with the model to improve the upscaling results.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`upscale_by`**
    - The factor by which the image will be upscaled, determining the increase in resolution.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`seed`**
    - A seed value for random number generation, ensuring reproducibility of the upscaling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps the model will take in enhancing the image, affecting the detail and quality of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - The classifier-free guidance scale, adjusting the influence of the conditioning texts on the upscaling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The name of the sampler used in the upscaling process, affecting the texture and details of the upscaled image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler for the sampling process, managing how the model iterates over the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - The amount of denoising applied during upscaling, affecting the smoothness and clarity of the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mode_type`**
    - The mode of operation for upscaling, which can include different strategies for handling image tiles and seams.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`tile_width`**
    - The width of the tiles used in the upscaling process, which can affect the performance and quality of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tile_height`**
    - The height of the tiles used in the upscaling process, similar to tile width in its impact on the upscaling.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mask_blur`**
    - The level of blur applied to the mask used in tiled upscaling, affecting the blending of tiles.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tile_padding`**
    - The padding added around each tile during upscaling, which helps in reducing visible seams between tiles.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seam_fix_mode`**
    - The method used for fixing seams between tiles, crucial for achieving a seamless upscale.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seam_fix_denoise`**
    - The amount of denoising applied specifically to the seams, improving the overall smoothness of the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`seam_fix_width`**
    - The width of the area around seams to be fixed, affecting the transition between tiles.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seam_fix_mask_blur`**
    - The level of blur applied to the seam fix mask, affecting how seamlessly the fixed seams blend with the rest of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seam_fix_padding`**
    - The padding around the seam fix area, used to further blend and smooth out the transitions between tiles.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`force_uniform_tiles`**
    - A flag indicating whether to use uniform tile sizes throughout the image, which can affect the consistency of the upscale.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`tiled_decode`**
    - Indicates whether the decoding process should be done in a tiled manner, affecting the handling of large images.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`upscale_model`**
    - Specifies the model used for upscaling, enabling the selection of different algorithms or models tailored to specific upscaling needs.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `str`
- **`custom_sampler`**
    - Defines a custom sampling method for the upscaling process, allowing for more control over the sampling techniques used during image enhancement.
    - Comfy dtype: `SAMPLER`
    - Python dtype: `str`
- **`custom_sigmas`**
    - Allows for the customization of sigma values used in the sampling process, offering fine-tuning of the noise levels applied during image upscaling.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `List[float]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an enhanced version of the input image, with improved resolution and detail quality through the upscaling process.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
# Built-in or C extension class, unable to automatically detect source code
```
