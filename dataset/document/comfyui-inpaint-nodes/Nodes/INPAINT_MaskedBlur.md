---
tags:
- Blur
- VisualEffects
---

# Blur Masked Area
## Documentation
- Class name: `INPAINT_MaskedBlur`
- Category: `inpaint`
- Output node: `False`

The node applies a blurring effect to areas of an image specified by a mask, with adjustable intensity and falloff, blending the altered regions smoothly with the rest of the image. It's designed for inpainting tasks where selective blurring can help integrate edited areas more naturally.
## Input types
### Required
- **`image`**
    - The input image to be processed. The mask defines areas to blur, allowing for selective focus or smoothing within the image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - A binary mask indicating areas of the image to apply the blur. It works in tandem with the image to selectively target regions for blurring.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`blur`**
    - Specifies the intensity of the blur to be applied. Higher values result in more pronounced blurring, affecting the visual softness of the masked areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`falloff`**
    - Determines the transition smoothness between blurred and unblurred areas, enabling a gradual blend that enhances the natural appearance of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is the modified image where specified areas have been blurred according to the mask and blur settings, seamlessly integrating edits.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskedBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "blur": ("INT", {"default": 255, "min": 3, "max": 8191, "step": 1}),
                "falloff": ("INT", {"default": 0, "min": 0, "max": 8191, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "inpaint"
    FUNCTION = "fill"

    def fill(self, image: Tensor, mask: Tensor, blur: int, falloff: int):
        blur = make_odd(blur)
        falloff = min(make_odd(falloff), blur - 2)
        image, mask = to_torch(image, mask)

        original = image.clone()
        alpha = mask_floor(mask)
        if falloff > 0:
            erosion = binary_erosion(alpha, falloff)
            alpha = alpha * gaussian_blur(erosion, falloff)
        alpha = alpha.expand(-1, 3, -1, -1)

        image = gaussian_blur(image, blur)
        image = original + (image - original) * alpha
        return (to_comfy(image),)

```
