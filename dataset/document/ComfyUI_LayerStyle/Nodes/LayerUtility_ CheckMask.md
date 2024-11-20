# LayerUtility: Check Mask
## Documentation
- Class name: `LayerUtility: CheckMask`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The CheckMask node is designed to evaluate the effectiveness of a given mask by determining if the area of the mask that meets certain criteria (e.g., white point value) exceeds a specified percentage. This functionality is crucial for filtering out ineffective masks in image processing workflows, ensuring that subsequent operations are performed on valid data.
## Input types
### Required
- **`mask`**
    - The mask input is the primary data on which the node operates, used to determine if the mask is effective based on its white point value and area coverage.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`white_point`**
    - The white point parameter sets the threshold value for considering a pixel as part of the effective area of the mask, playing a critical role in the mask's evaluation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`area_percent`**
    - The area percent parameter specifies the minimum percentage of the mask that must be considered effective, based on the white point criteria, for the mask to be deemed valid.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`bool`**
    - Comfy dtype: `BOOLEAN`
    - Indicates whether the mask is considered effective based on the white point and area percentage criteria.
    - Python dtype: `bool`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CheckMask:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        blank_mask_list = ['white', 'black']
        return {
            "required": {
                "mask": ("MASK",),  #
                "white_point": ("INT", {"default": 1, "min": 1, "max": 254, "step": 1}), # 用于判断mask是否有效的白点值，高于此值被计入有效
                "area_percent": ("INT", {"default": 1, "min": 1, "max": 99, "step": 1}), # 区域百分比，低于此则mask判定无效
            },
            "optional": { #
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ('bool',)
    FUNCTION = 'check_mask'
    CATEGORY = '😺dzNodes/LayerUtility'

    def check_mask(self, mask, white_point, area_percent,):

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        mask = tensor2pil(mask[0])
        if mask.width * mask.height > 262144:
            target_width = 512
            target_height = int(target_width * mask.height / mask.width)
            mask = mask.resize((target_width, target_height), Image.LANCZOS)
        return (mask_white_area(mask, white_point) * 100 > area_percent,)

```
