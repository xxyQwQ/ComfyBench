---
tags:
- Color
---

# Color Clip ADE20k
## Documentation
- Class name: `Color Clip ADE20k`
- Category: `Bmad/image`
- Output node: `False`

The Color Clip ADE20k node is designed to modify the color palette of an image based on the ADE20K dataset, allowing for the clipping of image colors to a specific class color from the dataset. It extends the functionality of a base color clipping operation by incorporating ADE20K-specific color mappings, enabling targeted color transformations for image processing tasks.
## Input types
### Required
- **`image`**
    - The input image to be processed. This image will undergo color clipping based on the specified class name and target color operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`target`**
    - Specifies the target operation for the color clipping, such as converting to black, white, or maintaining the color.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`complement`**
    - Defines the complementary operation for the color clipping, providing additional control over the color transformation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`class_name`**
    - The name of the class from the ADE20K dataset whose color will be used for clipping. This determines the specific color to which the image's colors will be clipped.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image with colors clipped according to the specified class color from the ADE20K dataset.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorClipADE20K(ColorClip):
    @classmethod
    def INPUT_TYPES(cls):
        types = super().get_types(advanced=False)
        types["required"].pop("color")
        types["required"]["class_name"] = (ade20k_class_names, {"default": default_class_name})
        return types

    def color_clip(self, image, class_name, target, complement):
        clip_color = list((ADE20K_dic[class_name] * 255).astype(uint8))
        # note: max eucl. dist. between 2 colors in the dictionary is 7.xxx ... w/ a diff of (4, 5, 3)
        image = self.clip(image, clip_color, target, complement, leeway=2)
        return (image,)

```
