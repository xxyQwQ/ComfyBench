---
tags:
- Mask
- MaskGeneration
---

# Offset Mask
## Documentation
- Class name: `OffsetMask`
- Category: `KJNodes/masking`
- Output node: `False`

The OffsetMask node is designed to manipulate and transform masks by applying specified offsets, rotations, and potentially duplicating the mask to create a batch with varied transformations. This node allows for dynamic adjustments to mask positioning and orientation, enabling more complex mask-based operations and effects within image processing workflows.
## Input types
### Required
- **`mask`**
    - The input mask or batch of masks to be transformed. It serves as the base for applying offsets, rotations, and duplications, directly influencing the output mask's appearance and arrangement.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`x`**
    - Specifies the horizontal offset to apply to the mask. A positive value shifts the mask right, while a negative value shifts it left, affecting the mask's horizontal position.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - Specifies the vertical offset to apply to the mask. A positive value shifts the mask down, while a negative value shifts it up, affecting the mask's vertical position.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`angle`**
    - The rotation angle in degrees to apply to the mask. This transformation rotates the mask around its center, altering its orientation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`duplication_factor`**
    - The number of times the input mask is duplicated to form a batch. This parameter allows for the creation of multiple transformed copies of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`roll`**
    - Determines whether edge wrapping (rolling) is applied during the offset. When true, the mask's edges wrap around, creating a continuous effect.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`incremental`**
    - When set to true, applies the horizontal and vertical offsets incrementally across the mask batch, creating a progressive transformation effect.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`padding_mode`**
    - Defines the padding mode used when applying offsets and rotations, affecting how the mask's edges are handled during transformation. Options include 'empty', 'border', and 'reflection'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The transformed mask or batch of masks after applying the specified offsets, rotations, and duplications. It reflects the cumulative effect of all transformations.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OffsetMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "x": ("INT", { "default": 0, "min": -4096, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "y": ("INT", { "default": 0, "min": -4096, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "angle": ("INT", { "default": 0, "min": -360, "max": 360, "step": 1, "display": "number" }),
                "duplication_factor": ("INT", { "default": 1, "min": 1, "max": 1000, "step": 1, "display": "number" }),
                "roll": ("BOOLEAN", { "default": False }),
                "incremental": ("BOOLEAN", { "default": False }),
                "padding_mode": (
            [   
                'empty',
                'border',
                'reflection',
                
            ], {
               "default": 'empty'
            }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "offset"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Offsets the mask by the specified amount.  
 - mask: Input mask or mask batch
 - x: Horizontal offset
 - y: Vertical offset
 - angle: Angle in degrees
 - roll: roll edge wrapping
 - duplication_factor: Number of times to duplicate the mask to form a batch
 - border padding_mode: Padding mode for the mask
"""

    def offset(self, mask, x, y, angle, roll=False, incremental=False, duplication_factor=1, padding_mode="empty"):
        # Create duplicates of the mask batch
        mask = mask.repeat(duplication_factor, 1, 1).clone()

        batch_size, height, width = mask.shape

        if angle != 0 and incremental:
            for i in range(batch_size):
                rotation_angle = angle * (i+1)
                mask[i] = TF.rotate(mask[i].unsqueeze(0), rotation_angle).squeeze(0)
        elif angle > 0:
            for i in range(batch_size):
                mask[i] = TF.rotate(mask[i].unsqueeze(0), angle).squeeze(0)

        if roll:
            if incremental:
                for i in range(batch_size):
                    shift_x = min(x*(i+1), width-1)
                    shift_y = min(y*(i+1), height-1)
                    if shift_x != 0:
                        mask[i] = torch.roll(mask[i], shifts=shift_x, dims=1)
                    if shift_y != 0:
                        mask[i] = torch.roll(mask[i], shifts=shift_y, dims=0)
            else:
                shift_x = min(x, width-1)
                shift_y = min(y, height-1)
                if shift_x != 0:
                    mask = torch.roll(mask, shifts=shift_x, dims=2)
                if shift_y != 0:
                    mask = torch.roll(mask, shifts=shift_y, dims=1)
        else:
            
            for i in range(batch_size):
                if incremental:
                    temp_x = min(x * (i+1), width-1)
                    temp_y = min(y * (i+1), height-1)
                else:
                    temp_x = min(x, width-1)
                    temp_y = min(y, height-1)
                if temp_x > 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([torch.zeros((height, temp_x)), mask[i, :, :-temp_x]], dim=1)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :, :-temp_x], (0, temp_x), mode=padding_mode)
                elif temp_x < 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([mask[i, :, :temp_x], torch.zeros((height, -temp_x))], dim=1)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :, -temp_x:], (temp_x, 0), mode=padding_mode)

                if temp_y > 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([torch.zeros((temp_y, width)), mask[i, :-temp_y, :]], dim=0)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :-temp_y, :], (0, temp_y), mode=padding_mode)
                elif temp_y < 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([mask[i, :temp_y, :], torch.zeros((-temp_y, width))], dim=0)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, -temp_y:, :], (temp_y, 0), mode=padding_mode)
           
        return mask,

```
