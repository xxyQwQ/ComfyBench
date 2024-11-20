---
tags:
- Color
---

# Color Clip
## Documentation
- Class name: `Color Clip`
- Category: `Bmad/image`
- Output node: `False`

The Color Clip node is designed to modify the colors within an image based on specific target and complement operations, optionally allowing for advanced color manipulation through additional color parameters. It leverages color theory principles to enhance or alter the visual appearance of images, making it a versatile tool for image processing tasks that require color adjustments.
## Input types
### Required
- **`image`**
    - The image to be processed. It serves as the primary input for color clipping operations, determining the visual content that will undergo color modifications.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`target`**
    - Specifies the target operation for color clipping, influencing how colors in the image are adjusted or replaced.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`complement`**
    - Defines the complement operation for color clipping, determining the alternative color adjustments or replacements in the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`color`**
    - The reference color used for clipping operations. It plays a crucial role in defining the color adjustments to be applied to the image.
    - Comfy dtype: `COLOR`
    - Python dtype: `List[int]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image with applied color clipping operations, showcasing the adjustments or alterations made to the original colors.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorClipSimple(ColorClip):
    @classmethod
    def INPUT_TYPES(cls):
        return super().get_types(advanced=False)

    def color_clip(self, image, color, target, complement):
        image = self.clip(image, color, target, complement)
        return (image,)

```
