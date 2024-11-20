---
tags:
- ImageFilter
- VisualEffects
---

# Guided Filter Image
## Documentation
- Class name: `GuidedFilterImage`
- Category: `image/filters`
- Output node: `False`

The GuidedFilterImage node applies a guided filter to a batch of images using a guide image to enhance the output images' details while preserving edges. This process involves adjusting the images based on the spatial correlations indicated by the guide image, making it suitable for tasks like smoothing, detail enhancement, or noise reduction.
## Input types
### Required
- **`images`**
    - The batch of images to be filtered. This input is crucial for determining the content that will undergo the guided filtering process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`guide`**
    - The guide image used to direct the filtering process. It plays a key role in determining how the filtering is applied across different regions of the input images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`size`**
    - Specifies the size of the kernel used for the guided filter. A larger size can lead to more pronounced smoothing and edge preservation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma`**
    - Controls the degree of smoothing in the guided filtering process. Higher values result in stronger smoothing effects.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The filtered images, which have undergone guided filtering to enhance details while preserving edges.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GuidedFilterImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "guide": ("IMAGE", ),
                "size": ("INT", {"default": 4, "min": 0, "max": 1023}),
                "sigma": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "filter_image"

    CATEGORY = "image/filters"

    def filter_image(self, images, guide, size, sigma):
        d = size * 2 + 1
        s = sigma / 10
        filtered = guided_filter_tensor(guide, images, d, s)
        return (filtered,)

```
