# VFI FloatToInt
## Documentation
- Class name: `VFI FloatToInt`
- Category: `ComfyUI-Frame-Interpolation`
- Output node: `False`

The VFI FloatToInt node is designed to convert floating-point numbers to integers. This conversion process is essential in scenarios where integer values are required for further processing or when interfacing with systems that only accept integers. The node can handle both single floating-point values and collections of values, offering flexibility in its application.
## Input types
### Required
- **`float`**
    - The floating-point number or collection of floating-point numbers to be converted to integer(s). This input is crucial for the conversion process, determining the output based on the value(s) provided.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float or Iterable[float]`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The converted integer or list of integers from the provided floating-point number(s). This output is significant for processes that require integer values.
    - Python dtype: `int or List[int]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FloatToInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float": ("FLOAT", {"default": 0, 'min': 0, 'step': 0.01})
            }
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "convert"
    CATEGORY = "ComfyUI-Frame-Interpolation"

    def convert(self, float):
        if hasattr(float, "__iter__"):
            return (list(map(int, float)),)
        return (int(float),)

```
