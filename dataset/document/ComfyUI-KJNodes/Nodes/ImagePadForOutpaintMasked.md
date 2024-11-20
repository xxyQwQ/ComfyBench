---
tags:
- Image
- ImagePadding
- ImageTransformation
---

# Image Pad For Outpaint Masked
## Documentation
- Class name: `ImagePadForOutpaintMasked`
- Category: `image`
- Output node: `False`

This node is designed to adjust the dimensions of an image for outpainting tasks by applying padding to the specified sides of the image. It extends the functionality of outpainting by allowing for masked areas, enabling selective padding and processing based on the mask provided. This is particularly useful in scenarios where the image composition needs to be carefully controlled or when integrating new elements into existing images without affecting certain regions.
## Input types
### Required
- **`image`**
    - The image to be processed and padded. It is the primary input on which the node operates, determining the base for padding adjustments.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`left`**
    - The amount of padding to apply to the left side of the image. This parameter directly influences the horizontal expansion of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`top`**
    - The amount of padding to apply to the top side of the image. It affects the vertical expansion of the image, particularly on the top edge.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`right`**
    - The calculated amount of padding to apply to the right side of the image, based on the target width, current width, and left padding. It adjusts the horizontal dimensions of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bottom`**
    - The amount of padding to apply to the bottom side of the image. This parameter influences the vertical expansion of the image on the bottom edge.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`feathering`**
    - The degree of feathering to apply along the edges of the padding. This parameter controls the smoothness of the transition between the original image and the padded areas.
    - Comfy dtype: `INT`
    - Python dtype: `bool`
### Optional
- **`mask`**
    - A scaled version of the mask that defines areas to be excluded or differently processed during the padding operation. It allows for selective padding adjustments.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an image with applied padding according to the specified parameters and mask, ready for further processing or outpainting tasks.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The mask output reflects the areas that were selectively processed or excluded during the padding operation, maintaining the integrity of the original masked regions.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImagePadForOutpaintMasked:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"

    def expand_image(self, image, left, top, right, bottom, feathering, mask=None):
        if mask is not None:
            if torch.allclose(mask, torch.zeros_like(mask)):
                    print("Warning: The incoming mask is fully black. Handling it as None.")
                    mask = None
        B, H, W, C = image.size()

        new_image = torch.ones(
            (B, H + top + bottom, W + left + right, C),
            dtype=torch.float32,
        ) * 0.5

        new_image[:, top:top + H, left:left + W, :] = image

        if mask is None:
            new_mask = torch.ones(
                (B, H + top + bottom, W + left + right),
                dtype=torch.float32,
            )

            t = torch.zeros(
            (B, H, W),
            dtype=torch.float32
            )
        else:
            # If a mask is provided, pad it to fit the new image size
            mask = F.pad(mask, (left, right, top, bottom), mode='constant', value=0)
            mask = 1 - mask
            t = torch.zeros_like(mask)
        
        if feathering > 0 and feathering * 2 < H and feathering * 2 < W:

            for i in range(H):
                for j in range(W):
                    dt = i if top != 0 else H
                    db = H - i if bottom != 0 else H

                    dl = j if left != 0 else W
                    dr = W - j if right != 0 else W

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    if mask is None:
                        t[:, i, j] = v * v
                    else:
                        t[:, top + i, left + j] = v * v
        
        if mask is None:
            new_mask[:, top:top + H, left:left + W] = t
            return (new_image, new_mask,)
        else:
            return (new_image, mask,)

```
