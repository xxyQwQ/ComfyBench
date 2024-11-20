# Remap Range
## Documentation
- Class name: `RemapRange`
- Category: `cspnodes`
- Output node: `False`

The RemapRange node is designed to adjust a given value from one range to another, optionally clamping the output within the target range. This functionality is essential for normalizing data or adapting values to fit within specific bounds, making it a versatile tool for data preprocessing and transformation.
## Input types
### Required
- **`value`**
    - Specifies the value to be remapped. Its role is crucial as it determines the base value that will undergo transformation according to the specified input and output ranges.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`input_min`**
    - Defines the lower bound of the original range. It sets the starting point from which the value will be scaled.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`input_max`**
    - Sets the upper limit of the original range. This parameter, in conjunction with input_min, establishes the range within which the value currently resides.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`output_min`**
    - Determines the lower boundary of the target range. It marks the minimum value to which the original value can be scaled.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`output_max`**
    - Specifies the upper boundary of the target range. This parameter defines the maximum potential value after remapping.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`clamp`**
    - A boolean flag that, when true, clamps the remapped value within the output range, ensuring it does not exceed the specified bounds.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The result of the remapping process. It represents the original value adjusted to fit within the new specified range.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapRange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "input_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "input_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "output_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "output_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "clamp": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "remap_value"
    CATEGORY = "cspnodes"

    def remap_value(self, value, input_min, input_max, output_min, output_max, clamp):
        # Calculate the input and output ranges
        input_range = input_max - input_min
        output_range = output_max - output_min
        
        # Perform the remapping
        if input_range == 0:
            remapped = output_min
        else:
            remapped = ((value - input_min) / input_range) * output_range + output_min
        
        # Clamp the output if requested
        if clamp:
            remapped = max(min(remapped, output_max), output_min)
        
        return (remapped,)

```
