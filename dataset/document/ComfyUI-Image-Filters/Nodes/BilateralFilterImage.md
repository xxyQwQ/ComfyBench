---
tags:
- ImageFilter
- VisualEffects
---

# Bilateral Filter Image
## Documentation
- Class name: `BilateralFilterImage`
- Category: `image/filters`
- Output node: `False`

The BilateralFilterImage node applies a bilateral filter to images, a process that reduces noise while preserving edges. This technique is particularly useful for creating smoother images without blurring important features, making it ideal for enhancing image quality in a nuanced manner.
## Input types
### Required
- **`images`**
    - The images to be filtered. This input is crucial for defining the visual content that will undergo the bilateral filtering process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`size`**
    - Specifies the diameter of the pixel neighborhood used for the filter. A larger size can lead to more significant smoothing and noise reduction.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma_color`**
    - Controls the filter's sensitivity to color differences. Higher values allow more colors to be mixed within the pixel neighborhood.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigma_space`**
    - Determines the spatial extent of the filter. Higher values mean that farther pixels will influence each other as long as their colors are close enough.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The filtered image, which has undergone bilateral filtering to reduce noise while preserving edges.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BilateralFilterImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "size": ("INT", {"default": 8, "min": 1, "max": 64}),
                "sigma_color": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "sigma_space": ("FLOAT", {"default": 100.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "filter_image"

    CATEGORY = "image/filters"

    def filter_image(self, images, size, sigma_color, sigma_space):
        np_images = images.detach().clone().cpu().numpy()
        d = size * 2 + 1
        for index, image in enumerate(np_images):
            np_images[index] = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return (torch.from_numpy(np_images),)

```
