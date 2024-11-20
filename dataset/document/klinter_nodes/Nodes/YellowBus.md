# YellowBus (klinter)
## Documentation
- Class name: `YellowBus`
- Category: `None`
- Output node: `False`

YellowBus is designed to facilitate the transfer of various types of data, including models, VAEs, conditioning prompts, and latent embeddings, without modification. It acts as a conduit, allowing these elements to flow seamlessly from input to output, thereby enabling the preservation and straightforward exchange of information between different stages of a processing pipeline.
## Input types
### Required
- **`model`**
    - Represents a model input, allowing for the transfer of model data through the node.
    - Comfy dtype: `MODEL`
    - Python dtype: `object`
- **`vae`**
    - Represents a VAE (Variational Autoencoder) input, facilitating the transfer of VAE data through the node.
    - Comfy dtype: `VAE`
    - Python dtype: `object`
- **`pos_prompt`**
    - Represents a positive conditioning prompt, enabling the transfer of specific conditioning information intended to positively influence the outcome.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`neg_prompt`**
    - Represents a negative conditioning prompt, enabling the transfer of specific conditioning information intended to negatively influence the outcome.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`latent`**
    - Represents latent embeddings input, allowing for the transfer of latent space representations through the node.
    - Comfy dtype: `LATENT`
    - Python dtype: `numpy.ndarray`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Outputs the model data received as input, unchanged.
    - Python dtype: `object`
- **`vae`**
    - Comfy dtype: `VAE`
    - Outputs the VAE (Variational Autoencoder) data received as input, unchanged.
    - Python dtype: `object`
- **`pos_prompt`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the positive conditioning prompt received as input, unchanged.
    - Python dtype: `str`
- **`neg_prompt`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the negative conditioning prompt received as input, unchanged.
    - Python dtype: `str`
- **`latent`**
    - Comfy dtype: `LATENT`
    - Outputs the latent embeddings received as input, unchanged.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class YellowBus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"forceInput": False,}),         # Input is a model
                "vae": ("VAE", {"forceInput": False,}),             # Input is a VAE
                "pos_prompt": ("CONDITIONING", {"forceInput": False,}),  # Input is a positive prompt (conditioning)
                "neg_prompt": ("CONDITIONING",  {"forceInput": False,}),  # Input is a negative prompt (conditioning)
                "latent": ("LATENT", {"forceInput": False,}),       # Input is latent embeddings
            }
        }

    RETURN_TYPES = ("MODEL", "VAE", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "vae", "pos_prompt", "neg_prompt", "latent")
    FUNCTION = "transfer"

    def transfer(self, model, vae, pos_prompt, neg_prompt, latent):
        # Return inputs directly as outputs
        return (model, vae, pos_prompt, neg_prompt, latent)

```
