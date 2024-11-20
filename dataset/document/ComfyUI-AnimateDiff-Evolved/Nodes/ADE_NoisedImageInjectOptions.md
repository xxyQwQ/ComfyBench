---
tags:
- Animation
---

# Image Injection Options 🎭🅐🅓
## Documentation
- Class name: `ADE_NoisedImageInjectOptions`
- Category: `Animate Diff 🎭🅐🅓/sample settings/image inject`
- Output node: `False`

This node is designed to create and configure options for injecting noised images into an animation sequence. It allows for the specification of positional adjustments for the injected images, enabling precise control over their placement within the animation frames.
## Input types
### Required
### Optional
- **`composite_x`**
    - Specifies the horizontal position offset for the image injection. This allows for fine-tuning the placement of the injected image within the frame.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`composite_y`**
    - Specifies the vertical position offset for the image injection. This enables precise vertical alignment of the injected image within the animation frame.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMG_INJECT_OPTS`**
    - Comfy dtype: `IMAGE_INJECT_OPTIONS`
    - Returns an instance of NoisedImageInjectOptions configured with the specified positional adjustments. This object is used to control the placement of noised images in the animation sequence.
    - Python dtype: `NoisedImageInjectOptions`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class NoisedImageInjectOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "composite_x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "composite_y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE_INJECT_OPTIONS",)
    RETURN_NAMES = ("IMG_INJECT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/image inject"
    FUNCTION = "create_image_inject_opts"

    def create_image_inject_opts(self, x=0, y=0):
        return (NoisedImageInjectOptions(x=x, y=y),)

```
