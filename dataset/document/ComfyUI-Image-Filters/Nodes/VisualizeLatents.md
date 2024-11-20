---
tags:
- Latent
---

# Visualize Latents
## Documentation
- Class name: `VisualizeLatents`
- Category: `utils`
- Output node: `False`

The VisualizeLatents node transforms latent space representations into visualizable images, adjusting their mean and standard deviation for better visual interpretation. It organizes the visualized latents into a grid layout based on the square root of the number of channels, facilitating the examination of latent space features.
## Input types
### Required
- **`latent`**
    - The 'latent' input is a dictionary containing the latent space samples to be visualized. It is crucial for generating the visual representation of the latent space, affecting the output image's appearance and layout.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an image tensor representing the visualized latent space, adjusted for better visual interpretation and organized in a grid layout.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class VisualizeLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT", ),}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"

    CATEGORY = "utils"

    def visualize(self, latent):
        latents = latent["samples"]
        batch, channels, height, width = latents.size()
        
        latents = latents - latents.mean()
        latents = latents / latents.std()
        latents = latents / 10 + 0.5
        
        scale = int(channels ** 0.5)
        vis = torch.zeros(batch, height * scale, width * scale)
        
        for i in range(channels):
            start_h  = (i % scale) * height
            end_h    = start_h + height
            start_w  = (i // scale) * width
            end_w    = start_w + width
            
            vis[:, start_h:end_h, start_w:end_w] = latents[:, i]
        
        return (vis.unsqueeze(-1).repeat(1, 1, 1, 3),)

```
