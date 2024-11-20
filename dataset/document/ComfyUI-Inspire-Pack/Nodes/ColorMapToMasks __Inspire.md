---
tags:
- Mask
---

# Color Map To Masks (Inspire)
## Documentation
- Class name: `ColorMapToMasks __Inspire`
- Category: `InspirePack/Util`
- Output node: `False`

The `ColorMapToMasks` node is designed to process a color map image to identify and extract masks corresponding to the top colors present in the image. It aims to simplify the segmentation of images into distinct color-based regions, facilitating further image manipulation or analysis tasks.
## Input types
### Required
- **`color_map`**
    - The `color_map` parameter represents the input image from which masks will be generated based on the dominant colors. It plays a crucial role in determining the segmentation outcome.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`min_pixels`**
    - The `min_pixels` parameter sets the minimum number of pixels a color must have within the color map to be considered for mask creation. This helps in filtering out less significant colors.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`max_count`**
    - The `max_count` parameter limits the number of top colors to be considered for mask generation. This controls the granularity of the segmentation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a tensor of masks, each corresponding to one of the top colors identified in the input color map. These masks enable targeted manipulation or analysis of specific color regions within the image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorMapToMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "color_map": ("IMAGE",),
                    "min_pixels": ("INT", {"default": 500, "min": 1, "max": 0xffffffffffffffff, "step": 1}),
                    "max_count": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
                    },
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, color_map, max_count, min_pixels):
        if len(color_map) > 0:
            print(f"[Inspire Pack] WARN: ColorMapToMasks - Sure, here's the translation: `color_map` can only be a single image. Only the first image will be processed. If you want to utilize the remaining images, convert the Image Batch to an Image List.")

        top_colors = top_k_colors(color_map[0], max_count, min_pixels)

        masks = None

        for color in top_colors:
            this_mask = create_mask(color_map, color)
            if masks is None:
                masks = this_mask
            else:
                masks = torch.concat((masks, this_mask), dim=0)

        if masks is None:
            masks = torch.zeros_like(color_map[0, :, :, 0])
            masks.unsqueeze(0)

        return (masks,)

```
