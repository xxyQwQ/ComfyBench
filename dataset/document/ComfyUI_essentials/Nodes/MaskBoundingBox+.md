---
tags:
- BoundingBox
- Image
- ImageTransformation
---

# 🔧 Mask Bounding Box
## Documentation
- Class name: `MaskBoundingBox+`
- Category: `essentials/mask`
- Output node: `False`

This node is designed for processing masks to identify and extract bounding box coordinates, facilitating tasks that require spatial localization within images. It can be used for cropping, object detection, and area-specific processing, enhancing the utility of masks by providing precise location data.
## Input types
### Required
- **`mask`**
    - The input mask for which the bounding box is to be computed, representing a specific area within an image. The node calculates the bounding box that tightly encloses this area, providing spatial localization.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`padding`**
    - An optional parameter that specifies the amount of padding to add to the bounding box, allowing for adjustments to the size of the extracted area.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - An optional parameter that applies a blur to the mask before computing the bounding box, potentially smoothing the edges and affecting the final bounding box coordinates.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`image_optional`**
    - An optional image input that, if provided, can be used in conjunction with the mask to refine the bounding box calculation based on the image content.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`MASK`**
    - Comfy dtype: `MASK`
    - The mask adjusted to the bounding box, potentially including padding and modifications from the blur parameter.
    - Python dtype: `torch.Tensor`
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - An optional output image cropped to the bounding box of the mask, provided if an image input was given.
    - Python dtype: `torch.Tensor`
- **`x`**
    - Comfy dtype: `INT`
    - The x-coordinate of the top-left corner of the bounding box.
    - Python dtype: `int`
- **`y`**
    - Comfy dtype: `INT`
    - The y-coordinate of the top-left corner of the bounding box.
    - Python dtype: `int`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the bounding box.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the bounding box.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskBoundingBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1, }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
            },
            "optional": {
                "image_optional": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("MASK", "IMAGE", "x", "y", "width", "height")
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, padding, blur, image_optional=None):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if image_optional is None:
            image_optional = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        # resize the image if it's not the same size as the mask
        if image_optional.shape[1:] != mask.shape[1:]:
            image_optional = comfy.utils.common_upscale(image_optional.permute([0,3,1,2]), mask.shape[2], mask.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        # match batch size
        if image_optional.shape[0] < mask.shape[0]:
            image_optional = torch.cat((image_optional, image_optional[-1].unsqueeze(0).repeat(mask.shape[0]-image_optional.shape[0], 1, 1, 1)), dim=0)
        elif image_optional.shape[0] > mask.shape[0]:
            image_optional = image_optional[:mask.shape[0]]

        # blur the mask
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            mask = T.functional.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1)

        _, y, x = torch.where(mask)
        x1 = max(0, x.min().item() - padding)
        x2 = min(mask.shape[2], x.max().item() + 1 + padding)
        y1 = max(0, y.min().item() - padding)
        y2 = min(mask.shape[1], y.max().item() + 1 + padding)

        # crop the mask
        mask = mask[:, y1:y2, x1:x2]
        image_optional = image_optional[:, y1:y2, x1:x2, :]

        return (mask, image_optional, x1, y1, x2 - x1, y2 - y1)

```
