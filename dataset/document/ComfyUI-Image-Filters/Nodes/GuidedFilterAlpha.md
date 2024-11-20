---
tags:
- AlphaChannel
- Image
---

# (DEPRECATED) Guided Filter Alpha
## Documentation
- Class name: `GuidedFilterAlpha`
- Category: `image/filters`
- Output node: `False`

The GuidedFilterAlpha node applies a guided filter to an image using a specified alpha mask. This process smooths the image while preserving edge details, guided by the alpha mask. It's designed for enhancing images or compositing tasks where maintaining edge integrity is crucial.
## Input types
### Required
- **`images`**
    - The input images to be filtered. This parameter is crucial for defining the visual content that will undergo the guided filtering process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`alpha`**
    - The alpha mask used to guide the filtering process. It influences how the filtering is applied, particularly around edges, to preserve details.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`filter_radius`**
    - Defines the radius of the filter. A larger radius increases the area of influence for smoothing, affecting the level of detail preservation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma`**
    - Controls the degree of smoothing. Higher values result in more aggressive smoothing, affecting the filter's sensitivity to edges.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The filtered image, where the guided filter has been applied according to the alpha mask and other parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GuidedFilterAlpha:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "alpha": ("IMAGE",),
                "filter_radius": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "guided_filter_alpha"

    CATEGORY = "image/filters"

    def guided_filter_alpha(self, images: torch.Tensor, alpha: torch.Tensor, filter_radius: int, sigma: float):
        
        d = filter_radius * 2 + 1
        s = sigma / 10
        
        i_dup = copy.deepcopy(images.cpu().numpy())
        a_dup = copy.deepcopy(alpha.cpu().numpy())
        
        for index, image in enumerate(i_dup):
            alpha_work = a_dup[index]
            i_dup[index] = guidedFilter(image, alpha_work, d, s)
        
        return (torch.from_numpy(i_dup),)

```
