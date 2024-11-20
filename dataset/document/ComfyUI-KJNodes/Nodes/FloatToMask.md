---
tags:
- Mask
- MaskGeneration
---

# Float To Mask
## Documentation
- Class name: `FloatToMask`
- Category: `KJNodes/masking/generate`
- Output node: `False`

The `FloatToMask` node is designed to convert floating-point representations into mask formats, facilitating operations that require binary or discrete mask inputs. This node plays a crucial role in image processing and manipulation tasks where precise delineation of areas based on specific criteria is essential.
## Input types
### Required
- **`input_values`**
    - Represents the floating-point values to be converted into a mask. These values are typically thresholds or indicators that determine the mask's shape and boundaries.
    - Comfy dtype: `FLOAT`
    - Python dtype: `torch.Tensor`
- **`width`**
    - Specifies the width of the output mask. This parameter determines the horizontal dimension of the mask generated from the input values.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the output mask. This parameter determines the vertical dimension of the mask generated from the input values.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output binary mask generated from the input floating-point values. This mask can be used for various image processing applications that require discrete delineation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FloatToMask:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_values": ("FLOAT", {"forceInput": True, "default": 0}),
                "width": ("INT", {"default": 100, "min": 1}),
                "height": ("INT", {"default": 100, "min": 1}),
            },
        }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes/masking/generate"
    DESCRIPTION = """
Generates a batch of masks based on the input float values.
The batch size is determined by the length of the input float values.
Each mask is generated with the specified width and height.
"""

    def execute(self, input_values, width, height):
        import pandas as pd
        # Ensure input_values is a list
        if isinstance(input_values, (float, int)):
            input_values = [input_values]
        elif isinstance(input_values, pd.Series):
            input_values = input_values.tolist()
        elif isinstance(input_values, list) and all(isinstance(item, list) for item in input_values):
            input_values = [item for sublist in input_values for item in sublist]

        # Generate a batch of masks based on the input_values
        masks = []
        for value in input_values:
            # Assuming value is a float between 0 and 1 representing the mask's intensity
            mask = torch.ones((height, width), dtype=torch.float32) * value
            masks.append(mask)
        masks_out = torch.stack(masks, dim=0)
    
        return(masks_out,)

```
