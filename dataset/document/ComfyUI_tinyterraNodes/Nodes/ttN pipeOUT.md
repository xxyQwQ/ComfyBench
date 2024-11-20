---
tags:
- Image
- Pipeline
- PipelineTransformation
---

# pipeOUT (Legacy)
## Documentation
- Class name: `ttN pipeOUT`
- Category: `🌏 tinyterra/legacy`
- Output node: `False`

The ttN pipeOUT node is designed for the final stage of a pipeline in the tinyterra legacy system, where it extracts and returns various components of a given pipeline such as models, conditioning data, latent representations, and more. This node serves as a crucial endpoint for retrieving processed data from the pipeline.
## Input types
### Required
- **`pipe`**
    - The 'pipe' parameter represents the pipeline from which data is to be extracted. It is essential for the operation of the node, as it contains all the information needed for the output.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the model component extracted from the pipeline.
    - Python dtype: `str`
- **`pos`**
    - Comfy dtype: `CONDITIONING`
    - Returns the positive conditioning data extracted from the pipeline.
    - Python dtype: `str`
- **`neg`**
    - Comfy dtype: `CONDITIONING`
    - Returns the negative conditioning data extracted from the pipeline.
    - Python dtype: `str`
- **`latent`**
    - Comfy dtype: `LATENT`
    - Returns the latent representation data extracted from the pipeline.
    - Python dtype: `list`
- **`vae`**
    - Comfy dtype: `VAE`
    - Returns the VAE component extracted from the pipeline.
    - Python dtype: `str`
- **`clip`**
    - Comfy dtype: `CLIP`
    - Returns the CLIP component extracted from the pipeline.
    - Python dtype: `str`
- **`image`**
    - Comfy dtype: `IMAGE`
    - Returns the image data extracted from the pipeline.
    - Python dtype: `list`
- **`seed`**
    - Comfy dtype: `INT`
    - Returns the seed used in the pipeline's process.
    - Python dtype: `int`
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Returns the entire pipeline, allowing for further manipulation or analysis.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipe_OUT:
    version = '1.1.0'
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                },
            "hidden": {"ttNnodeVersion": ttN_pipe_OUT.version},
            }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT", "PIPE_LINE",)
    RETURN_NAMES = ("model", "pos", "neg", "latent", "vae", "clip", "image", "seed", "pipe")
    FUNCTION = "flush"

    CATEGORY = "🌏 tinyterra/legacy"
    
    def flush(self, pipe):
        model = pipe.get("model")
        pos = pipe.get("positive")
        neg = pipe.get("negative")
        latent = pipe.get("samples")
        vae = pipe.get("vae")
        clip = pipe.get("clip")
        image = pipe.get("images")
        seed = pipe.get("seed")

        return model, pos, neg, latent, vae, clip, image, seed, pipe

```
