# LayerMask: MaskMotionBlur
## Documentation
- Class name: `LayerMask: MaskMotionBlur`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The MaskMotionBlur node applies a motion blur effect to a mask, optionally inverting the mask before processing. It allows for the customization of the blur intensity and direction, enabling dynamic visual effects on mask elements.
## Input types
### Required
- **`mask`**
    - The mask to which the motion blur effect will be applied. It serves as the primary input for the transformation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - A boolean flag that determines whether the input mask should be inverted before applying the motion blur effect. Inversion can highlight or alter areas of interest.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blur`**
    - Specifies the intensity of the motion blur effect. Higher values result in a more pronounced blur, simulating faster motion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`angle`**
    - Defines the angle of the motion blur effect, allowing for directional blurring that simulates motion in a specific direction.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The resulting mask after applying the motion blur effect, potentially inverted based on the input flag.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskMotionBlur:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blur": ("INT", {"default": 20, "min": 1, "max": 9999, "step": 1}),
                "angle": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_motion_blur'
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_motion_blur(self, mask, invert_mask, blur, angle,):

        l_masks = []
        ret_masks = []

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)

        for m in mask:
            if invert_mask:
                m = 1 - m
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_masks)):
            _mask = l_masks[i]
            _blurimage = motion_blur(_mask, angle, blur)
            ret_masks.append(image2mask(_blurimage))

        log(f"{NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)

```
