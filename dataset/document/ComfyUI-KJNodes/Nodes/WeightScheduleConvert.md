---
tags:
- DataTypeConversion
- NumericConversion
---

# Weight Schedule Convert
## Documentation
- Class name: `WeightScheduleConvert`
- Category: `KJNodes/weights`
- Output node: `False`

The WeightScheduleConvert node is designed for the conversion of value lists or series into different types, accommodating various data formats for further processing. It enables the transformation of input data to a specified output type, including options like matching the input format, converting to lists, pandas series, or tensors. This functionality is crucial for ensuring data compatibility and optimizing performance across different stages of a computational workflow.
## Input types
### Required
- **`input_values`**
    - Specifies the initial set of values to be converted. Its format and type are crucial for determining the appropriate conversion process and the resulting output type.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`output_type`**
    - Determines the desired format of the output, allowing for conversion to formats such as 'match_input', 'list', 'pandas series', or 'tensor'. This choice affects how the input values are transformed and represented in the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`invert`**
    - A boolean flag that, when set, may alter the conversion process by inverting certain aspects of the input data, depending on the specific implementation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`repeat`**
    - An integer specifying how many times the input values should be repeated in the output. This parameter can be used to expand the size of the output dataset.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`remap_to_frames`**
    - An optional integer parameter that specifies if and how the input values should be remapped to a specific number of frames, affecting the temporal dimension of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`interpolation_curve`**
    - An optional floating-point parameter that defines the curve used for interpolating between input values, influencing the smoothness and dynamics of the conversion process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`remap_values`**
    - A boolean flag indicating whether the input values should be remapped according to specified minimum and maximum values, enabling normalization or other forms of value adjustment.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`remap_min`**
    - An optional floating-point parameter specifying the minimum value to which input values can be remapped, used in conjunction with 'remap_max' to define the range of the remapping process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`remap_max`**
    - An optional floating-point parameter specifying the maximum value to which input values can be remapped, used in conjunction with 'remap_min' to define the range of the remapping process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The output can include floating-point numbers, representing transformed or converted values based on the specified output type.
    - Python dtype: `float`
- **`string`**
    - Comfy dtype: `STRING`
    - The output may contain strings, particularly when the output type is set to formats like 'list' or 'pandas series' that involve textual representation.
    - Python dtype: `str`
- **`int`**
    - Comfy dtype: `INT`
    - The output can include integers, which may result from certain types of conversions or processing applied to the input values.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WeightScheduleConvert:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_values": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "output_type": (
                [   
                    'match_input',
                    'list',
                    'pandas series',
                    'tensor',
                ],
                {
                "default": 'list'
                    }),
                "invert": ("BOOLEAN", {"default": False}),
                "repeat": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
             },
             "optional": {
                "remap_to_frames": ("INT", {"default": 0}),
                "interpolation_curve": ("FLOAT", {"forceInput": True}),
                "remap_values": ("BOOLEAN", {"default": False}),
                "remap_min": ("FLOAT", {"default": 0.0, "min": -100000, "max": 100000.0, "step": 0.01}),
                "remap_max": ("FLOAT", {"default": 1.0, "min": -100000, "max": 100000.0, "step": 0.01}),
             },
             
        }
    RETURN_TYPES = ("FLOAT", "STRING", "INT",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes/weights"
    DESCRIPTION = """
Converts different value lists/series to another type.  
"""

    def detect_input_type(self, input_values):
        import pandas as pd
        if isinstance(input_values, list):
            return 'list'
        elif isinstance(input_values, pd.Series):
            return 'pandas series'
        elif isinstance(input_values, torch.Tensor):
            return 'tensor'
        else:
            raise ValueError("Unsupported input type")

    def execute(self, input_values, output_type, invert, repeat, remap_to_frames=0, interpolation_curve=None, remap_min=0.0, remap_max=1.0, remap_values=False):
        import pandas as pd
        input_type = self.detect_input_type(input_values)

        if input_type == 'pandas series':
            float_values = input_values.tolist()
        elif input_type == 'tensor':
            float_values = input_values
        else:
            float_values = input_values

        if invert:
            float_values = [1 - value for value in float_values]

        if interpolation_curve is not None:
            interpolated_pattern = []
            orig_float_values = float_values
            for value in interpolation_curve:
                min_val = min(orig_float_values)
                max_val = max(orig_float_values)
                # Normalize the values to [0, 1]
                normalized_values = [(value - min_val) / (max_val - min_val) for value in orig_float_values]
                # Interpolate the normalized values to the new frame count
                remapped_float_values = np.interp(np.linspace(0, 1, int(remap_to_frames * value)), np.linspace(0, 1, len(normalized_values)), normalized_values).tolist()
                interpolated_pattern.extend(remapped_float_values)
            float_values = interpolated_pattern
        else:
            # Remap float_values to match target_frame_amount
            if remap_to_frames > 0 and remap_to_frames != len(float_values):
                min_val = min(float_values)
                max_val = max(float_values)
                # Normalize the values to [0, 1]
                normalized_values = [(value - min_val) / (max_val - min_val) for value in float_values]
                # Interpolate the normalized values to the new frame count
                float_values = np.interp(np.linspace(0, 1, remap_to_frames), np.linspace(0, 1, len(normalized_values)), normalized_values).tolist()
       
            float_values = float_values * repeat
            if remap_values:
                float_values = self.remap_values(float_values, remap_min, remap_max)

        if output_type == 'list':
            out = float_values,
        elif output_type == 'pandas series':
            out = pd.Series(float_values),
        elif output_type == 'tensor':
            if input_type == 'pandas series':
                out = torch.tensor(float_values.values, dtype=torch.float32),
            else:   
                out = torch.tensor(float_values, dtype=torch.float32),
        elif output_type == 'match_input':
            out = float_values,
        return (out, [str(value) for value in float_values], [int(value) for value in float_values])
    
    def remap_values(self, values, target_min, target_max):
        # Determine the current range
        current_min = min(values)
        current_max = max(values)
        current_range = current_max - current_min
        
        # Determine the target range
        target_range = target_max - target_min
        
        # Perform the linear interpolation for each value
        remapped_values = [(value - current_min) / current_range * target_range + target_min for value in values]
        
        return remapped_values

```
