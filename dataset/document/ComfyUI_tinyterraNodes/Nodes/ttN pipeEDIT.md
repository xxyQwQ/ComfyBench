---
tags:
- Image
- Pipeline
---

# pipeEDIT
## Documentation
- Class name: `ttN pipeEDIT`
- Category: `🌏 tinyterra/pipe`
- Output node: `False`

The ttN pipeEDIT node is designed for editing and refining the parameters and settings of a pipeline, allowing for the customization and optimization of the pipeline's behavior and output. It focuses on enhancing the flexibility and control over the pipeline's execution by enabling adjustments to its components and configurations.
## Input types
### Required
### Optional
- **`pipe`**
    - Represents the pipeline to be edited or refined, serving as the core structure for modification. It is crucial for determining the base configuration and elements that will undergo adjustments.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`model`**
    - Specifies the model component of the pipeline, allowing for its selection or modification.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`pos`**
    - Defines the positive conditioning aspect of the pipeline, enabling its specification or alteration.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`neg`**
    - Specifies the negative conditioning component, allowing for adjustments to how the pipeline handles negative inputs.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`latent`**
    - Represents the latent space configuration of the pipeline, permitting adjustments to its handling of latent variables.
    - Comfy dtype: `LATENT`
    - Python dtype: `str`
- **`vae`**
    - Specifies the VAE (Variational Autoencoder) component, enabling its selection or modification within the pipeline.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`clip`**
    - Defines the CLIP model component of the pipeline, allowing for its specification or alteration.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`image`**
    - Specifies the image component of the pipeline, enabling adjustments to how images are handled.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`seed`**
    - Represents the seed value for random number generation within the pipeline, allowing for its specification to ensure reproducibility.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Returns the modified pipeline, reflecting the applied edits and refinements.
    - Python dtype: `dict`
- **`model`**
    - Comfy dtype: `MODEL`
    - Outputs the selected or modified model component of the pipeline.
    - Python dtype: `str`
- **`pos`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the specified or altered positive conditioning component of the pipeline.
    - Python dtype: `str`
- **`neg`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the adjusted negative conditioning component of the pipeline.
    - Python dtype: `str`
- **`latent`**
    - Comfy dtype: `LATENT`
    - Returns the adjusted latent space configuration of the pipeline.
    - Python dtype: `str`
- **`vae`**
    - Comfy dtype: `VAE`
    - Outputs the selected or modified VAE component of the pipeline.
    - Python dtype: `str`
- **`clip`**
    - Comfy dtype: `CLIP`
    - Outputs the specified or altered CLIP model component of the pipeline.
    - Python dtype: `str`
- **`image`**
    - Comfy dtype: `IMAGE`
    - Outputs the adjusted image component of the pipeline.
    - Python dtype: `str`
- **`seed`**
    - Comfy dtype: `INT`
    - Returns the specified seed value for random number generation within the pipeline.
    - Python dtype: `int`
- **`ui`**
    - Provides a user interface component, typically for displaying information or results derived from the node's processing.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipe_EDIT:
    version = '1.1.1'
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {},
                "optional": {
                    "pipe": ("PIPE_LINE",),
                    "model": ("MODEL",),
                    "pos": ("CONDITIONING",),
                    "neg": ("CONDITIONING",),
                    "latent": ("LATENT",),
                    "vae": ("VAE",),
                    "clip": ("CLIP",),
                    "image": ("IMAGE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                },
                "hidden": {"ttNnodeVersion": ttN_pipe_EDIT.version, "my_unique_id": "UNIQUE_ID"},
            }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT")
    RETURN_NAMES = ("pipe", "model", "pos", "neg", "latent", "vae", "clip", "image", "seed")
    FUNCTION = "flush"

    CATEGORY = "🌏 tinyterra/pipe"

    def flush(self, pipe=None, model=None, pos=None, neg=None, latent=None, vae=None, clip=None, image=None, seed=None, my_unique_id=None):

        model = model or pipe.get("model")
        if model is None:
            ttNl("Model missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        pos = pos or pipe.get("positive")
        if pos is None:
            ttNl("Positive conditioning missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        neg = neg or pipe.get("negative")
        if neg is None:
            ttNl("Negative conditioning missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        samples = latent or pipe.get("samples")
        if samples is None:
            ttNl("Latent missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        vae = vae or pipe.get("vae")
        if vae is None:
            ttNl("VAE missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        clip = clip or pipe.get("clip")
        if clip is None:
            ttNl("Clip missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        image = image or pipe.get("images")
        if image is None:
            ttNl("Image missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        seed = seed or pipe.get("seed")
        if seed is None:
            ttNl("Seed missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()

        new_pipe = {
            "model": model,
            "positive": pos,
            "negative": neg,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": image,
            "seed": seed,

            "loader_settings": pipe["loader_settings"],
        }
        del pipe

        return (new_pipe, model, pos, neg, latent, vae, clip, image, seed)

```
