---
tags:
- ImageTransformation
- VisualEffects
---

# 🔧 Image Posterize
## Documentation
- Class name: `ImagePosterize+`
- Category: `essentials/image processing`
- Output node: `False`

The ImagePosterize node is designed for image processing tasks, specifically to posterize images. It simplifies the color palette of an image based on a threshold, effectively reducing the number of colors to create a stylized effect. This node operates on the principle of comparing pixel values against a threshold and adjusting them accordingly to achieve the posterization effect.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the input image to be posterized. It is crucial for defining the visual content that will undergo the posterization process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`threshold`**
    - The 'threshold' parameter determines the cutoff value for posterization. Pixels above this threshold will be treated differently from those below it, playing a key role in defining the posterization effect's intensity and appearance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a posterized version of the input image, where the color palette has been simplified according to the specified threshold.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImagePosterize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", { "default": 0.50, "min": 0.00, "max": 1.00, "step": 0.05, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, threshold):
        image = image.mean(dim=3, keepdim=True)
        image = (image > threshold).float()
        image = image.repeat(1, 1, 1, 3)

        return(image,)

```
