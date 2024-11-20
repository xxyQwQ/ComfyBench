---
tags:
- Mask
- MaskGeneration
---

# Resize Mask
## Documentation
- Class name: `ResizeMask`
- Category: `KJNodes/masking`
- Output node: `False`

The ResizeMask node is designed to adjust the dimensions of a given mask or a batch of masks to a specified width and height, optionally maintaining the original proportions. This functionality is crucial for tasks that require masks to match the dimensions of corresponding images or other layers within a graphical or machine learning pipeline.
## Input types
### Required
- **`mask`**
    - The mask parameter represents the input mask or batch of masks that will be resized. It is central to the node's operation, determining the base data that will be transformed.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`width`**
    - Specifies the target width for the resized mask. If keep_proportions is True and width is set to 0, the original width is maintained, adjusting only the height based on the aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the target height for the resized mask. Similar to width, if keep_proportions is True and height is set to 0, the original height is preserved, adjusting the width to maintain the aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`keep_proportions`**
    - A boolean flag indicating whether to maintain the original aspect ratio of the mask during resizing. When True, the mask is resized to fit within the specified dimensions without altering its aspect ratio.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The resized mask or batch of masks, adjusted to the specified dimensions.
    - Python dtype: `torch.Tensor`
- **`width`**
    - Comfy dtype: `INT`
    - The actual width of the resized mask, which may differ from the input width if keep_proportions is True.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The actual height of the resized mask, which may differ from the input height if keep_proportions is True.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ResizeMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, "display": "number" }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, "display": "number" }),
                "keep_proportions": ("BOOLEAN", { "default": False }),
            }
        }

    RETURN_TYPES = ("MASK", "INT", "INT",)
    RETURN_NAMES = ("mask", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Resizes the mask or batch of masks to the specified width and height.
"""

    def resize(self, mask, width, height, keep_proportions):
        if keep_proportions:
            _, oh, ow = mask.shape
            width = ow if width == 0 else width
            height = oh if height == 0 else height
            ratio = min(width / ow, height / oh)
            width = round(ow*ratio)
            height = round(oh*ratio)
        outputs = mask.unsqueeze(1)
        outputs = F.interpolate(outputs, size=(height, width), mode="nearest")
        outputs = outputs.squeeze(1)

        return(outputs, outputs.shape[2], outputs.shape[1],)

```
