---
tags:
- Image
- Pipeline
- PipelineTransformation
---

# pipeIN (Legacy)
## Documentation
- Class name: `ttN pipeIN`
- Category: `🌏 tinyterra/legacy`
- Output node: `False`

The ttN pipeIN node is designed for initializing and configuring pipelines within the tinyterra ecosystem, specifically tailored for legacy applications. It focuses on setting up the initial parameters required for a pipeline, including models, conditioning, latent space, VAE, CLIP, and seed values, along with optional image inputs.
## Input types
### Required
- **`model`**
    - Specifies the model to be used in the pipeline, serving as the foundational component for subsequent operations.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`pos`**
    - Defines positive conditioning inputs to guide the model's generation towards desired attributes or features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`neg`**
    - Specifies negative conditioning inputs to steer the model's generation away from certain attributes or features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`latent`**
    - Determines the latent space dimensions to be explored during generation, affecting the diversity and novelty of outputs.
    - Comfy dtype: `LATENT`
    - Python dtype: `str`
- **`vae`**
    - Specifies the VAE (Variational Autoencoder) to be used for generating or refining outputs, contributing to the quality and variability of the results.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`clip`**
    - Defines the CLIP model to be used for semantic understanding and alignment of generated content with textual descriptions.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`seed`**
    - Sets the seed value for random number generation, ensuring reproducibility of results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`image`**
    - Optional parameter for including an image input to be used in conjunction with other inputs for generation or refinement.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Outputs a configured pipeline object ready for further processing or generation within the tinyterra ecosystem.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipe_IN:
    version = '1.1.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },"optional": {
                "image": ("IMAGE",),
            },
            "hidden": {"ttNnodeVersion": ttN_pipe_IN.version},
        }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe", )
    FUNCTION = "flush"

    CATEGORY = "🌏 tinyterra/legacy"

    def flush(self, model, pos=0, neg=0, latent=0, vae=0, clip=0, image=0, seed=0):
        pipe = {"model": model,
                "positive": pos,
                "negative": neg,
                "vae": vae,
                "clip": clip,

                "refiner_model": None,
                "refiner_positive": None,
                "refiner_negative": None,
                "refiner_vae": None,
                "refiner_clip": None,

                "samples": latent,
                "images": image,
                "seed": seed,

                "loader_settings": {}
        }
        return (pipe, )

```
