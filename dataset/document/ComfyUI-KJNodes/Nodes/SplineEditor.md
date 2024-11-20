---
tags:
- XYPlotData
---

# Spline Editor
## Documentation
- Class name: `SplineEditor`
- Category: `KJNodes/weights`
- Output node: `False`

The SplineEditor node is a graphical tool designed for creating and editing splines that represent various schedules or mask batches. It provides a user-friendly interface for manipulating control points and curves, enabling the customization of values for different applications. This node is particularly useful for tasks that require precise control over the interpolation and distribution of values across a specified domain, such as animation scheduling or data visualization.
## Input types
### Required
- **`points_store`**
    - A storage mechanism for the control points. It plays a critical role in managing and accessing the points for spline generation and manipulation.
    - Comfy dtype: `STRING`
    - Python dtype: `object`
- **`coordinates`**
    - A list of coordinates for control points. These points define the shape and trajectory of the spline, directly impacting its form and the interpolated values.
    - Comfy dtype: `STRING`
    - Python dtype: `list`
- **`mask_width`**
    - Specifies the width of the mask to be generated. It determines the horizontal dimension of the output mask or batch, affecting the resolution and scale of the generated spline.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mask_height`**
    - Defines the height of the mask to be generated. It influences the vertical dimension of the output, playing a crucial role in determining the resolution and scale of the spline.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`points_to_sample`**
    - Sets the number of sample points to be generated from the spline. This value determines the granularity of the output, impacting the precision and detail of the interpolated values.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sampling_method`**
    - Chooses the method for sampling points along the spline, either by time or path. This affects how values are distributed along the spline, catering to different use cases like schedules or coordinates.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`interpolation`**
    - Specifies the method of interpolation between control points. This choice influences the smoothness and curvature of the spline, affecting its overall appearance and functionality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`tension`**
    - Controls the tension of the spline curve. Adjusting this parameter affects the curvature and tightness of the spline, allowing for finer control over its shape.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`repeat_output`**
    - Determines whether the output should be repeated. This can be useful for creating looping animations or patterns that require a continuous sequence of values.
    - Comfy dtype: `INT`
    - Python dtype: `bool`
- **`float_output_type`**
    - Determines the type of float values to be outputted, affecting the format and usability of the generated data for subsequent nodes or processes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`min_value`**
    - Sets the minimum value for the output range. This parameter ensures that the generated values do not fall below a certain threshold, maintaining consistency and control over the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_value`**
    - Establishes the maximum value for the output range. It caps the generated values to prevent them from exceeding a specified limit, ensuring the output remains within desired parameters.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - Generates a batch of masks based on the defined spline. This output is suitable for nodes that require mask inputs for further processing.
    - Python dtype: `object`
- **`coord_str`**
    - Comfy dtype: `STRING`
    - Produces a list of float values interpolated from the spline. These values can be used for weight adjustments or other numerical applications.
    - Python dtype: `list`
- **`float`**
    - Comfy dtype: `FLOAT`
    - Outputs a pandas series containing values derived from the spline. This format is convenient for data analysis and manipulation within pandas-supported environments.
    - Python dtype: `pandas.Series`
- **`count`**
    - Comfy dtype: `INT`
    - Creates a torch tensor from the spline's interpolated values. This output is compatible with PyTorch-based models and operations, facilitating integration into deep learning workflows.
    - Python dtype: `torch.Tensor`
- **`normalized_str`**
    - Comfy dtype: `STRING`
    - Provides a normalized string representation of the spline's interpolated values, useful for textual descriptions or further string-based processing.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SplineEditor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points_store": ("STRING", {"multiline": False}),
                "coordinates": ("STRING", {"multiline": False}),
                "mask_width": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "mask_height": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "points_to_sample": ("INT", {"default": 16, "min": 2, "max": 1000, "step": 1}),
                "sampling_method": (
                [   
                    'path',
                    'time',
                ],
                {
                    "default": 'time'
                }),
                "interpolation": (
                [   
                    'cardinal',
                    'monotone',
                    'basis',
                    'linear',
                    'step-before',
                    'step-after',
                    'polar',
                    'polar-reverse',
                ],
                {
                "default": 'cardinal'
                    }),
                "tension": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_output": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "float_output_type": (
                [   
                    'list',
                    'pandas series',
                    'tensor',
                ],
                {
                    "default": 'list'
                }),
            },
            "optional": {
                "min_value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "FLOAT", "INT", "STRING",)
    RETURN_NAMES = ("mask", "coord_str", "float", "count", "normalized_str",)
    FUNCTION = "splinedata"
    CATEGORY = "KJNodes/weights"
    DESCRIPTION = """
# WORK IN PROGRESS  
Do not count on this as part of your workflow yet,  
probably contains lots of bugs and stability is not  
guaranteed!!  
  
## Graphical editor to create values for various   
## schedules and/or mask batches.  

**Shift + click** to add control point at end.
**Ctrl + click** to add control point (subdivide) between two points.  
**Right click on a point** to delete it.    
Note that you can't delete from start/end.  
  
Right click on canvas for context menu:  
These are purely visual options, doesn't affect the output:  
 - Toggle handles visibility
 - Display sample points: display the points to be returned.  

**points_to_sample** value sets the number of samples  
returned from the **drawn spline itself**, this is independent from the  
actual control points, so the interpolation type matters.  
sampling_method: 
 - time: samples along the time axis, used for schedules  
 - path: samples along the path itself, useful for coordinates  

output types:
 - mask batch  
        example compatible nodes: anything that takes masks  
 - list of floats
        example compatible nodes: IPAdapter weights  
 - pandas series
        example compatible nodes: anything that takes Fizz'  
        nodes Batch Value Schedule  
 - torch tensor  
        example compatible nodes: unknown
"""

    def splinedata(self, mask_width, mask_height, coordinates, float_output_type, interpolation, 
                   points_to_sample, sampling_method, points_store, tension, repeat_output, min_value=0.0, max_value=1.0):
        
        coordinates = json.loads(coordinates)
        normalized = []
        normalized_y_values = []
        for coord in coordinates:
            coord['x'] = int(round(coord['x']))
            coord['y'] = int(round(coord['y']))
            norm_x = (1.0 - (coord['x'] / mask_height) - 0.0) * (max_value - min_value) + min_value
            norm_y = (1.0 - (coord['y'] / mask_height) - 0.0) * (max_value - min_value) + min_value
            normalized_y_values.append(norm_y)
            normalized.append({'x':norm_x, 'y':norm_y})
        if float_output_type == 'list':
            out_floats = normalized_y_values * repeat_output
        elif float_output_type == 'pandas series':
            try:
                import pandas as pd
            except:
                raise Exception("MaskOrImageToWeight: pandas is not installed. Please install pandas to use this output_type")
            out_floats = pd.Series(normalized_y_values * repeat_output),
        elif float_output_type == 'tensor':
            out_floats = torch.tensor(normalized_y_values * repeat_output, dtype=torch.float32)
        # Create a color map for grayscale intensities
        color_map = lambda y: torch.full((mask_height, mask_width, 3), y, dtype=torch.float32)

        # Create image tensors for each normalized y value
        mask_tensors = [color_map(y) for y in normalized_y_values]
        masks_out = torch.stack(mask_tensors)
        masks_out = masks_out.repeat(repeat_output, 1, 1, 1)
        masks_out = masks_out.mean(dim=-1)
        return (masks_out, json.dumps(coordinates), out_floats, len(out_floats) , json.dumps(normalized))

```
