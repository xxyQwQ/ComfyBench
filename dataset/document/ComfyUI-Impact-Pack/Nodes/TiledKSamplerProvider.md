---
tags:
- SamplerScheduler
- Sampling
---

# TiledKSamplerProvider
## Documentation
- Class name: `TiledKSamplerProvider`
- Category: `ImpactPack/Sampler`
- Output node: `False`

The TiledKSamplerProvider node is designed to facilitate the use of tiled sampling techniques within the ComfyUI framework. It dynamically integrates with the 'TiledKSampler' custom node, ensuring compatibility and extending functionality for advanced image generation tasks. This node acts as a bridge, enabling users to leverage tiled sampling strategies for improved performance and quality in image synthesis.
## Input types
### Required
- **`seed`**
    - Determines the random seed used for generating CPU noise for sampling, ensuring reproducibility and consistency in the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Specifies the total number of sampling steps to be executed, directly influencing the depth and detail of the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Sets the classifier free guidance value, adjusting the influence of conditioning on the sampling process for targeted image generation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Selects the specific sampler algorithm to be used, affecting the sampling behavior and output characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Specifies the scheduler for controlling the sampling process, impacting the progression and quality of image generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Adjusts the level of denoising applied during the sampling process, affecting the clarity and sharpness of the generated images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tile_width`**
    - Sets the width of the tiles used in the sampling process, influencing the granularity and detail of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tile_height`**
    - Sets the height of the tiles used in the sampling process, influencing the granularity and detail of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tiling_strategy`**
    - Determines the strategy for tiling during the sampling process, affecting the overall composition and coherence of the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`basic_pipe`**
    - A crucial input comprising the model and conditioning texts, which are essential for the sampling process. It determines the characteristics of the generated images by providing the necessary context and parameters for targeted image synthesis.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
## Output types
- **`ksampler`**
    - Comfy dtype: `KSAMPLER`
    - Outputs a sampler wrapper, which can be utilized for generating regional prompts in advanced image generation tasks.
    - Python dtype: `KSamplerWrapper`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class TiledKSamplerProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "padded", 'simple'], ),
                    "basic_pipe": ("BASIC_PIPE", )
                    }}

    TOOLTIPS = {
        "input": {
            "seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "cfg": "classifier free guidance value",
            "sampler_name": "sampler",
            "scheduler": "noise schedule",
            "denoise": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned.",
            "tile_width": "Sets the width of the tile to be used in TiledKSampler.",
            "tile_height": "Sets the height of the tile to be used in TiledKSampler.",
            "tiling_strategy": "Sets the tiling strategy for TiledKSampler.",
            "basic_pipe": "basic_pipe input for sampling",
        },
        "output": ("sampler wrapper. (Can be used when generating a regional_prompt.)", )
    }

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(seed, steps, cfg, sampler_name, scheduler, denoise,
             tile_width, tile_height, tiling_strategy, basic_pipe):
        model, _, _, positive, negative = basic_pipe
        sampler = core.TiledKSamplerWrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                                            tile_width, tile_height, tiling_strategy)
        return (sampler, )

```
