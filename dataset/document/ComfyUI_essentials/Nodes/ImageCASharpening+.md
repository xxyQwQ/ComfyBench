---
tags:
- ImageEnhancement
- VisualEffects
---

# 🔧 Image Contrast Adaptive Sharpening
## Documentation
- Class name: `ImageCASharpening+`
- Category: `essentials/image processing`
- Output node: `False`

The ImageCASharpening+ node applies contrast adaptive sharpening to images, enhancing their clarity and detail without significantly altering the overall composition. This process adjusts the sharpness of an image based on its local contrast, making it particularly effective for images that may appear soft or slightly out of focus.
## Input types
### Required
- **`image`**
    - The input image to be sharpened. This is the primary data upon which the contrast adaptive sharpening algorithm operates.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`amount`**
    - A floating-point value that determines the intensity of the sharpening effect. Higher values result in more pronounced sharpening, allowing for fine-tuning based on the specific needs of the image or application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying contrast adaptive sharpening. This image will have enhanced sharpness and detail, particularly in areas of varying contrast.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [ImageScaleBy](../../Comfy/Nodes/ImageScaleBy.md)
    - [ImpactSimpleDetectorSEGS_for_AD](../../ComfyUI-Impact-Pack/Nodes/ImpactSimpleDetectorSEGS_for_AD.md)



## Source code
```python
class ImageCAS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "essentials/image processing"
    FUNCTION = "execute"

    def execute(self, image, amount):
        epsilon = 1e-5
        img = F.pad(image.permute([0,3,1,2]), pad=(1, 1, 1, 1))

        a = img[..., :-2, :-2]
        b = img[..., :-2, 1:-1]
        c = img[..., :-2, 2:]
        d = img[..., 1:-1, :-2]
        e = img[..., 1:-1, 1:-1]
        f = img[..., 1:-1, 2:]
        g = img[..., 2:, :-2]
        h = img[..., 2:, 1:-1]
        i = img[..., 2:, 2:]

        # Computing contrast
        cross = (b, d, e, f, h)
        mn = min_(cross)
        mx = max_(cross)

        diag = (a, c, g, i)
        mn2 = min_(diag)
        mx2 = max_(diag)
        mx = mx + mx2
        mn = mn + mn2

        # Computing local weight
        inv_mx = torch.reciprocal(mx + epsilon)
        amp = inv_mx * torch.minimum(mn, (2 - mx))

        # scaling
        amp = torch.sqrt(amp)
        w = - amp * (amount * (1/5 - 1/8) + 1/8)
        div = torch.reciprocal(1 + 4*w)

        output = ((b + d + f + h)*w + e) * div
        output = output.clamp(0, 1)
        #output = torch.nan_to_num(output)

        output = output.permute([0,2,3,1])

        return (output,)

```
