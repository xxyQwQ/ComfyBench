---
tags:
- Image
- ImageBlend
- ImageComposite
---

# SeamlessClone
## Documentation
- Class name: `SeamlessClone`
- Category: `Bmad/CV/C.Photography`
- Output node: `False`

The SeamlessClone node is designed for blending two images together in a seamless manner, using various cloning modes to achieve different visual effects. It allows for precise control over the blending process by specifying the source and destination images, a mask to define the blend area, and the position for blending, along with the choice of cloning mode.
## Input types
### Required
- **`dst`**
    - The destination image onto which the source image will be cloned. It serves as the backdrop for the cloning operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`src`**
    - The source image to be cloned onto the destination image. This image will be blended into the destination based on the specified mask and cloning mode.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`src_mask`**
    - A mask image that defines the area of the source image to be cloned. Only the parts of the source image covered by the mask will be blended into the destination.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`flag`**
    - Specifies the cloning mode to be used for blending the images. The available modes are NORMAL, MIXED, and MONOCHROME_TRANSFER, each offering a different visual effect.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`cx`**
    - The x-coordinate of the center point in the destination image where the source image will be cloned.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cy`**
    - The y-coordinate of the center point in the destination image where the source image will be cloned.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The result of the seamless cloning operation, which is a blend of the source and destination images according to the specified parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SeamlessClone:
    clone_modes_map = {
        "NORMAL": cv.NORMAL_CLONE,
        "MIXED": cv.MIXED_CLONE,
        "MONO": cv.MONOCHROME_TRANSFER
    }
    clone_modes = list(clone_modes_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dst": ("IMAGE",),
                "src": ("IMAGE",),
                "src_mask": ("IMAGE",),
                "flag": (cls.clone_modes, {"default": cls.clone_modes[0]}),
                "cx": ("INT", {"default": 0, "min": -999999, "step": 1}),
                "cy": ("INT", {"default": 0, "min": -999999, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"
    CATEGORY = f"{cv_category_path}/C.Photography"

    def paste(self, src, dst, src_mask, flag, cx, cy):
        src = tensor2opencv(src)
        dst = tensor2opencv(dst)
        src_mask = tensor2opencv(src_mask, 1)

        result = cv.seamlessClone(src, dst, src_mask, (cx, cy), self.clone_modes_map[flag])
        result = opencv2tensor(result)

        return (result,)

```
