---
tags:
- LatentNoise
- Noise
---

# Generate Noise
## Documentation
- Class name: `GenerateNoise`
- Category: `KJNodes/noise`
- Output node: `False`

The GenerateNoise node is designed to create noise patterns that can be used to augment or modify latent representations in generative models. It provides a flexible way to generate noise with specific characteristics, such as size, seed, and intensity, allowing for controlled variations in the generated content.
## Input types
### Required
- **`width`**
    - Determines the width of the generated noise pattern, enabling customization of the noise to fit specific dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the generated noise pattern, allowing for the generation of noise that matches the desired dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_size`**
    - Specifies the number of noise samples to generate, allowing for batch processing of multiple noise patterns simultaneously.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed`**
    - Controls the randomness of the noise generation, ensuring reproducibility of the noise patterns when the same seed is used.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Adjusts the intensity of the generated noise, providing a means to amplify or reduce the noise effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`constant_batch_noise`**
    - When enabled, uses the same noise pattern across all samples in a batch, ensuring uniform noise characteristics.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`normalize`**
    - Normalizes the generated noise, ensuring that its distribution has a standard deviation of 1.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`model`**
    - Optional parameter that, if provided, can influence the noise generation process based on the model's characteristics, such as adjusting the noise to align with specific model requirements or enhancing compatibility with the model's latent space.
    - Comfy dtype: `MODEL`
    - Python dtype: `Optional[torch.nn.Module]`
- **`sigmas`**
    - Optional parameter that allows for scaling the noise based on the difference between two sigma values, typically used in diffusion models. This scaling adjusts the intensity and variability of the noise, directly influencing the characteristics of the generated patterns.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `Optional[List[float]]`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The generated noise pattern, ready to be used in further processing or as part of a generative model's input.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GenerateNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "multiplier": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 4096, "step": 0.01}),
            "constant_batch_noise": ("BOOLEAN", {"default": False}),
            "normalize": ("BOOLEAN", {"default": False}),
            },
            "optional": {
            "model": ("MODEL", ),
            "sigmas": ("SIGMAS", ),
            }
            }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generatenoise"
    CATEGORY = "KJNodes/noise"
    DESCRIPTION = """
Generates noise for injection or to be used as empty latents on samplers with add_noise off.
"""
        
    def generatenoise(self, batch_size, width, height, seed, multiplier, constant_batch_noise, normalize, sigmas=None, model=None):

        generator = torch.manual_seed(seed)
        noise = torch.randn([batch_size, 4, height // 8, width // 8], dtype=torch.float32, layout=torch.strided, generator=generator, device="cpu")
        if sigmas is not None:
            sigma = sigmas[0] - sigmas[-1]
            sigma /= model.model.latent_format.scale_factor
            noise *= sigma

        noise *=multiplier

        if normalize:
            noise = noise / noise.std()
        if constant_batch_noise:
            noise = noise[0].repeat(batch_size, 1, 1, 1)
        return ({"samples":noise}, )

```
