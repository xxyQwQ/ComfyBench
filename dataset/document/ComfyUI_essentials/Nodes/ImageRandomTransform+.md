---
tags:
- Image
- ImageTransformation
---

# 🔧 Image Random Transform
## Documentation
- Class name: `ImageRandomTransform+`
- Category: `essentials/image manipulation`
- Output node: `False`

This node applies a series of random transformations to an image, including perspective distortion, rotation, color jitter (brightness, contrast, saturation, hue), horizontal flipping, and random cropping. These transformations introduce variability and can be used to augment images for training or to generate diverse variations of a given image.
## Input types
### Required
- **`image`**
    - The input image to be transformed. It serves as the base for applying the random transformations, affecting the visual appearance and geometry of the output image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`seed`**
    - A seed value for the random number generator to ensure reproducibility of the transformations applied to the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`repeat`**
    - Specifies how many times the input image should be repeated before applying the transformations, effectively controlling the number of transformed images generated.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`variation`**
    - A scalar that modulates the intensity of the transformations applied, influencing the degree of variation in the output images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The transformed images, each having undergone a unique set of random transformations based on the specified parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageRandomTransform:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "repeat": ("INT", { "default": 1, "min": 1, "max": 256, "step": 1, }),
                "variation": ("FLOAT", { "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, seed, repeat, variation):
        h, w = image.shape[1:3]
        image = image.repeat(repeat, 1, 1, 1).permute([0, 3, 1, 2])

        distortion = 0.2 * variation
        rotation = 5 * variation
        brightness = 0.5 * variation
        contrast = 0.5 * variation
        saturation = 0.5 * variation
        hue = 0.2 * variation
        scale = 0.5 * variation

        torch.manual_seed(seed)

        out = []
        for i in image:
            tramsforms = T.Compose([
                T.RandomPerspective(distortion_scale=distortion, p=0.5),
                T.RandomRotation(degrees=rotation, interpolation=T.InterpolationMode.BILINEAR, expand=True),
                T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=(-hue, hue)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomResizedCrop((h, w), scale=(1-scale, 1+scale), ratio=(w/h, w/h), interpolation=T.InterpolationMode.BICUBIC),
            ])
            out.append(tramsforms(i.unsqueeze(0)))

        out = torch.cat(out, dim=0).permute([0, 2, 3, 1])

        return (out,)

```
