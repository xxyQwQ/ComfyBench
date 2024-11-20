---
tags:
- ImageTransformation
- VisualEffects
---

# 🔧 Pixelize
## Documentation
- Class name: `PixelOEPixelize+`
- Category: `essentials/image processing`
- Output node: `False`

The PixelOEPixelize node focuses on transforming images by applying a pixelization effect. It abstracts the complexity of manipulating pixel data to achieve a stylized reduction in image resolution, often used for aesthetic purposes or to anonymize sensitive information in images.
## Input types
### Required
- **`image`**
    - The input image to be pixelized. This image undergoes a transformation where its resolution is reduced in a stylized manner, effectively 'pixelizing' the image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`downscale_mode`**
    - Specifies the method used for downscaling the image as part of the pixelization process. Different modes can affect the visual outcome of the pixelization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`target_size`**
    - Defines the target size for the pixelization process, determining the final dimensions of the pixelized image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`patch_size`**
    - The size of each 'pixel' in the pixelized image, affecting the granularity of the pixelization effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`thickness`**
    - Controls the thickness of the grid lines in the pixelized image, adding to the stylized effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`color_matching`**
    - Determines how colors are matched during the pixelization process, influencing the color accuracy of the pixelized image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`upscale`**
    - A flag indicating whether the pixelized image should be upscaled back to its original resolution, affecting the final appearance.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after applying the pixelization effect, which may be upscaled to match the original image's resolution.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PixelOEPixelize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_mode": (["contrast", "bicubic", "nearest", "center", "k-centroid"],),
                "target_size": ("INT", { "default": 128, "min": 0, "max": MAX_RESOLUTION, "step": 8 }),
                "patch_size": ("INT", { "default": 16, "min": 4, "max": 32, "step": 2 }),
                "thickness": ("INT", { "default": 2, "min": 1, "max": 16, "step": 1 }),
                "color_matching": ("BOOLEAN", { "default": True }),
                "upscale": ("BOOLEAN", { "default": True }),
                #"contrast": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                #"saturation": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, downscale_mode, target_size, patch_size, thickness, color_matching, upscale):
        from pixeloe.pixelize import pixelize

        image = image.clone().mul(255).clamp(0, 255).byte().cpu().numpy()
        output = []
        for img in image:
            img = pixelize(img,
                           mode=downscale_mode,
                           target_size=target_size,
                           patch_size=patch_size,
                           thickness=thickness,
                           contrast=1.0,
                           saturation=1.0,
                           color_matching=color_matching,
                           no_upscale=not upscale)
            output.append(T.ToTensor()(img))

        output = torch.stack(output, dim=0).permute([0, 2, 3, 1])

        return(output,)

```
