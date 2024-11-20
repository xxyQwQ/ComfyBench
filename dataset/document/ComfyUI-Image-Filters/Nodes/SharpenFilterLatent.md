# Sharpen Filter (Latent)
## Documentation
- Class name: `SharpenFilterLatent`
- Category: `latent/filters`
- Output node: `False`

The SharpenFilterLatent node applies a sharpening filter to latent representations of images, enhancing their details by adjusting the contrast of edges within the latent space. This process involves blurring the latent image and then combining it with the original to accentuate edges, controlled by a specified factor and filter size.
## Input types
### Required
- **`latents`**
    - The latent representation of an image to be sharpened. This input is crucial for determining the base content that will undergo the sharpening process.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`filter_size`**
    - Specifies the size of the filter used for blurring the latent image before sharpening. A larger size leads to a more pronounced blurring effect, which can affect the sharpness intensity.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`factor`**
    - Controls the intensity of the sharpening effect applied to the latent image. A higher factor increases the contrast between edges and non-edges, making the image appear sharper.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The sharpened latent representation of the image, with enhanced edge details and contrast.
    - Python dtype: `Tuple[Dict[str, torch.Tensor]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SharpenFilterLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT", ),
                "filter_size": ("INT", {"default": 1, "min": 1, "max": 128}),
                "factor": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,  "round": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "filter_latent"

    CATEGORY = "latent/filters"

    def filter_latent(self, latents, filter_size, factor):
        latents_copy = copy.deepcopy(latents)
        t = latents_copy["samples"].movedim(1, -1) # [B x C x H x W] -> [B x H x W x C]
        
        d = filter_size * 2 + 1
        t_blurred = cv_blur_tensor(t, d, d)
        
        t = t - t_blurred
        t = t * factor
        t = t + t_blurred
        
        t = t.movedim(-1, 1) # [B x H x W x C] -> [B x C x H x W]
        latents_copy["samples"] = t
        return (latents_copy,)

```
