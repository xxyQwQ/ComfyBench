---
tags:
- ImageSize
- ImageTransformation
---

# Match Dimensions (mtb)
## Documentation
- Class name: `Match Dimensions (mtb)`
- Category: `mtb/utils`
- Output node: `False`

The MTB_MatchDimensions node is designed to adjust the dimensions of source images to match those of reference images along a specified dimension (height or width), ensuring the aspect ratio is preserved. This functionality is crucial for tasks requiring uniform image dimensions without distorting the image content.
## Input types
### Required
- **`source`**
    - The source image whose dimensions are to be adjusted. It plays a critical role in determining the new dimensions while preserving the aspect ratio.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`reference`**
    - The reference image provides the target dimensions to match. It sets the benchmark for adjusting the source image, ensuring consistency in dimensions across images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`match`**
    - Specifies the dimension (height or width) along which the source image's dimensions should be matched to the reference image. It guides the adjustment process, ensuring the aspect ratio is preserved.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The adjusted image with dimensions matched to the reference image, preserving the aspect ratio.
    - Python dtype: `IMAGE`
- **`new_width`**
    - Comfy dtype: `INT`
    - The new width of the adjusted image after matching dimensions.
    - Python dtype: `INT`
- **`new_height`**
    - Comfy dtype: `INT`
    - The new height of the adjusted image after matching dimensions.
    - Python dtype: `INT`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_MatchDimensions:
    """Match images dimensions along the given dimension, preserving aspect ratio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "reference": ("IMAGE",),
                "match": (["height", "width"], {"default": "height"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "new_width", "new_height")
    CATEGORY = "mtb/utils"
    FUNCTION = "execute"

    def execute(
        self, source: torch.Tensor, reference: torch.Tensor, match: str
    ):
        import torchvision.transforms.functional as VF

        _batch_size, height, width, _channels = source.shape
        _rbatch_size, rheight, rwidth, _rchannels = reference.shape

        source_aspect_ratio = width / height
        # reference_aspect_ratio = rwidth / rheight

        source = source.permute(0, 3, 1, 2)
        reference = reference.permute(0, 3, 1, 2)

        if match == "height":
            new_height = rheight
            new_width = int(rheight * source_aspect_ratio)
        else:
            new_width = rwidth
            new_height = int(rwidth / source_aspect_ratio)

        resized_images = [
            VF.resize(
                source[i],
                (new_height, new_width),
                antialias=True,
                interpolation=Image.BICUBIC,
            )
            for i in range(_batch_size)
        ]
        resized_source = torch.stack(resized_images, dim=0)
        resized_source = resized_source.permute(0, 2, 3, 1)

        return (resized_source, new_width, new_height)

```
