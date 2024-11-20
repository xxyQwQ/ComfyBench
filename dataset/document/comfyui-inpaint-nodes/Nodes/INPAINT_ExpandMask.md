# Expand Mask
## Documentation
- Class name: `INPAINT_ExpandMask`
- Category: `inpaint`
- Output node: `False`

The ExpandMask node is designed for modifying the input mask by expanding its boundaries and applying a blur effect, making it suitable for inpainting tasks where adjusting the mask size and softness can help in achieving better results.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter represents the input mask to be expanded and blurred. It is crucial for defining the area to be inpainted.
    - Comfy dtype: `MASK`
    - Python dtype: `Tensor`
- **`grow`**
    - The 'grow' parameter specifies the amount by which the mask's boundaries should be expanded. It plays a key role in determining the new size of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - The 'blur' parameter controls the intensity of the blur effect applied to the expanded mask, affecting the softness of its edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a modified mask with expanded boundaries and applied blur, ready for inpainting tasks.
    - Python dtype: `Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ExpandMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "grow": ("INT", {"default": 16, "min": 0, "max": 8096, "step": 1}),
                "blur": ("INT", {"default": 7, "min": 0, "max": 8096, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    CATEGORY = "inpaint"
    FUNCTION = "expand"

    def expand(self, mask: Tensor, grow: int, blur: int):
        mask = mask_unsqueeze(mask)
        if grow > 0:
            mask = binary_dilation(mask, grow)
        if blur > 0:
            mask = gaussian_blur(mask, make_odd(blur))
        return (mask.squeeze(1),)

```
