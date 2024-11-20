---
tags:
- Conditioning
---

# IC-Light Conditioning
## Documentation
- Class name: `ICLightConditioning`
- Category: `IC-Light`
- Output node: `False`

The ICLightConditioning node is designed to prepare and condition inputs specifically for the IC-Light model, ensuring compatibility and optimal performance. It focuses on adjusting and validating input data, such as images and noise, to meet the model's requirements, facilitating enhanced image generation processes.
## Input types
### Required
- **`positive`**
    - A set of conditioning instructions intended to guide the image generation towards desired attributes or features, enhancing the positive aspects of the output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict]]`
- **`negative`**
    - A set of conditioning instructions aimed at steering the image generation away from undesired attributes or features, minimizing the negative aspects of the output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict]]`
- **`vae`**
    - The variational autoencoder used to encode images into a latent space, providing a compressed representation of the input images for further processing.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`foreground`**
    - The latent representation of the foreground image to be used in the conditioning process, contributing to the generation of the final image.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`multiplier`**
    - A scalar value that adjusts the intensity of the effect of the foreground latent representation on the conditioning process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`opt_background`**
    - An optional latent representation of the background image, which can be used to modify the background of the generated image.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor] or None`
## Output types
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The modified set of positive conditioning instructions, incorporating the effects of the input latent representations and adjustments.
    - Python dtype: `List[Tuple[torch.Tensor, Dict]]`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The modified set of negative conditioning instructions, incorporating the effects of the input latent representations and adjustments.
    - Python dtype: `List[Tuple[torch.Tensor, Dict]]`
- **`empty_latent`**
    - Comfy dtype: `LATENT`
    - A placeholder latent representation, typically used when no specific output latent is required for the operation.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ICLightConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "foreground": ("LATENT", ),
                             "multiplier": ("FLOAT", {"default": 0.18215, "min": 0.0, "max": 1.0, "step": 0.001}),
                             },
                "optional": {
                     "opt_background": ("LATENT", ),
                     },
                }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("positive", "negative", "empty_latent")
    FUNCTION = "encode"
    CATEGORY = "IC-Light"
    DESCRIPTION = """
  
Conditioning for the IC-Light model.  
To use the "opt_background" input, you also need to use the  
"fbc" version of the IC-Light models.  
  
"""

    def encode(self, positive, negative, vae, foreground, multiplier, opt_background=None):
        samples_1 = foreground["samples"]

        if opt_background is not None:
            samples_2 = opt_background["samples"]

            repeats_1 = samples_2.size(0) // samples_1.size(0)
            repeats_2 = samples_1.size(0) // samples_2.size(0)
            if samples_1.shape[1:] != samples_2.shape[1:]:
                samples_2 = comfy.utils.common_upscale(samples_2, samples_1.shape[-1], samples_1.shape[-2], "bilinear", "disabled")

            # Repeat the tensors to match the larger batch size
            if repeats_1 > 1:
                samples_1 = samples_1.repeat(repeats_1, 1, 1, 1)
            if repeats_2 > 1:
                samples_2 = samples_2.repeat(repeats_2, 1, 1, 1)

            concat_latent = torch.cat((samples_1, samples_2), dim=1)
        else:
            concat_latent = samples_1

        out_latent = torch.zeros_like(samples_1)

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = concat_latent * multiplier
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], {"samples": out_latent})

```
