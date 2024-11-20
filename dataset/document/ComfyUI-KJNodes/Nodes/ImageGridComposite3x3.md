---
tags:
- GridLayout
- Image
- Tiled
---

# Image Grid Composite 3x3
## Documentation
- Class name: `ImageGridComposite3x3`
- Category: `KJNodes/image`
- Output node: `False`

This node is designed to concatenate nine input images into a single 3x3 grid image. It's particularly useful for creating composite images or visual summaries where multiple images need to be displayed together in a structured format.
## Input types
### Required
- **`image1`**
    - The first image to be placed in the top-left corner of the 3x3 grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image2`**
    - The second image to be placed in the top row, middle position of the 3x3 grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image3`**
    - The third image to be placed in the top row, right position of the 3x3 grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image4`**
    - The fourth image to be placed in the middle row, left position of the 3x3 grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image5`**
    - The central image of the 3x3 grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image6`**
    - The sixth image to be placed in the middle row, right position of the 3x3 grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image7`**
    - The seventh image to be placed in the bottom row, left position of the 3x3 grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image8`**
    - The eighth image to be placed in the bottom row, middle position of the 3x3 grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image9`**
    - The ninth image to be placed in the bottom row, right position of the 3x3 grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a single image that combines the nine input images into a 3x3 grid format.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageGridComposite3x3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "image3": ("IMAGE",),
            "image4": ("IMAGE",),
            "image5": ("IMAGE",),
            "image6": ("IMAGE",),
            "image7": ("IMAGE",),
            "image8": ("IMAGE",),
            "image9": ("IMAGE",),     
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compositegrid"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Concatenates the 9 input images into a 3x3 grid. 
"""

    def compositegrid(self, image1, image2, image3, image4, image5, image6, image7, image8, image9):
        top_row = torch.cat((image1, image2, image3), dim=2)
        mid_row = torch.cat((image4, image5, image6), dim=2)
        bottom_row = torch.cat((image7, image8, image9), dim=2)
        grid = torch.cat((top_row, mid_row, bottom_row), dim=1)
        return (grid,)

```
