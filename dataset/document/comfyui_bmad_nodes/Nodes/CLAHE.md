---
tags:
- Color
- ImageEnhancement
---

# CLAHE
## Documentation
- Class name: `CLAHE`
- Category: `Bmad/CV/Thresholding`
- Output node: `False`

The CLAHE node applies the Contrast Limited Adaptive Histogram Equalization algorithm to enhance the contrast of images. It adjusts image intensities to enhance the contrast, particularly useful for images with poor lighting conditions or to bring out more detail in specific areas.
## Input types
### Required
- **`src`**
    - The source image to be processed. It is the primary input for the contrast enhancement operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clip_limit`**
    - Defines the threshold for contrast limiting. A higher value increases contrast but may also amplify noise.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`tile_grid_x`**
    - Specifies the number of tiles in the horizontal direction for the adaptive histogram equalization. Affects the granularity of contrast enhancement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tile_grid_y`**
    - Specifies the number of tiles in the vertical direction for the adaptive histogram equalization. Affects the granularity of contrast enhancement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying CLAHE, with enhanced contrast.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CLAHE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                "clip_limit": ("INT", {"default": 2, "step": 1}),
                # 40 is the default in documentation, but prob. a bit high no?
                "tile_grid_x": ("INT", {"default": 8, "min": 2, "step": 1}),
                "tile_grid_y": ("INT", {"default": 8, "min": 2, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "eq"
    CATEGORY = f"{cv_category_path}/Thresholding"

    def eq(self, src, clip_limit, tile_grid_x, tile_grid_y):
        src = tensor2opencv(src, 1)
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_x, tile_grid_y))
        eq = clahe.apply(src)
        eq = cv.cvtColor(eq, cv.COLOR_GRAY2RGB)
        eq = opencv2tensor(eq)
        return (eq,)

```
