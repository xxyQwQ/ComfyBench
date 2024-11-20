---
tags:
- Color
---

# Color Clip (advanced)
## Documentation
- Class name: `Color Clip (advanced)`
- Category: `Bmad/image`
- Output node: `False`

The Color Clip (advanced) node is designed for advanced color manipulation within images, allowing for precise color clipping and replacement based on a set of advanced parameters. It extends basic color clipping capabilities by offering additional control over the color adjustment process, enabling users to specify alternative colors and adjust the leeway for color matching.
## Input types
### Required
- **`image`**
    - The input image to be processed for color clipping and manipulation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`target`**
    - Specifies the target operation for the primary color within the image, such as converting to black, white, or doing nothing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`complement`**
    - Defines the operation for colors in the image that do not match the primary color, including converting them or leaving them unchanged.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`color`**
    - The primary color to be clipped or replaced in the image.
    - Comfy dtype: `COLOR`
    - Python dtype: `List[int]`
### Optional
- **`color_a`**
    - An optional color parameter that provides an alternative color option for the clipping process.
    - Comfy dtype: `COLOR`
    - Python dtype: `List[int]`
- **`color_b`**
    - A second optional color parameter that offers another alternative color choice for the clipping process.
    - Comfy dtype: `COLOR`
    - Python dtype: `List[int]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The image after applying the advanced color clipping and manipulation operations.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorClipAdvanced(ColorClip):
    @classmethod
    def INPUT_TYPES(cls):
        return super().get_types(advanced=True)

    def color_clip(self, image, color, target, complement, color_a=None, color_b=None):
        image = self.clip(image, color, target, complement, color_a, color_b)
        return (image,)

```
