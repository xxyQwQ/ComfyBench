---
tags:
- Latent
---

# NNLatentUpscale
## Documentation
- Class name: `NNLatentUpscale`
- Category: `latent`
- Output node: `False`

The NNLatentUpscale node specializes in enhancing the resolution of SDXL latent representations using a neural network approach. It dynamically adjusts its internal model based on the specified version and upscale factor, ensuring optimized upscaling of latent samples to achieve higher fidelity outputs.
## Input types
### Required
- **`latent`**
    - The latent representation to be upscaled. It is crucial for defining the starting point of the upscaling process.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`version`**
    - Specifies the version of the model to use for upscaling, allowing for flexibility and optimization based on the latent's characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`upscale`**
    - The factor by which to upscale the latent representation, directly influencing the output's resolution and detail.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The upscaled latent representation, enhanced in resolution and detail.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [SamplerCustom](../../Comfy/Nodes/SamplerCustom.md)
    - [LatentInterpolate](../../Comfy/Nodes/LatentInterpolate.md)



## Source code
```python
class NNLatentUpscale:
    """
    Upscales SDXL latent using neural network
    """

    def __init__(self):
        self.local_dir = os.path.dirname(os.path.realpath(__file__))
        self.scale_factor = 0.13025
        self.dtype = torch.float32
        if model_management.should_use_fp16():
            self.dtype = torch.float16
        self.weight_path = {
            "SDXL": os.path.join(self.local_dir, "sdxl_resizer.pt"),
            "SD 1.x": os.path.join(self.local_dir, "sd15_resizer.pt"),
        }
        self.version = "none"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "version": (["SDXL", "SD 1.x"],),
                "upscale": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(self, latent, version, upscale):
        device = model_management.get_torch_device()
        samples = latent["samples"].to(device=device, dtype=self.dtype)

        if version != self.version:
            self.model = LatentResizer.load_model(
                self.weight_path[version], device, self.dtype
            )
            self.version = version

        self.model.to(device=device)
        latent_out = (
            self.model(self.scale_factor * samples, scale=upscale) / self.scale_factor
        )

        if self.dtype != torch.float32:
            latent_out = latent_out.to(dtype=torch.float32)

        latent_out = latent_out.to(device="cpu")

        self.model.to(device=model_management.vae_offload_device())
        return ({"samples": latent_out},)

```
