---
tags:
- Scheduling
---

# Convert Schedule List
## Documentation
- Class name: `SaltScheduleConverter`
- Category: `SALT/AudioViz/Scheduling`
- Output node: `False`

The SaltScheduleConverter node is designed to transform schedule lists into formats compatible with various modules by converting list elements into floats or integers. This ensures interoperability with modules that require specific numeric types, facilitating seamless integration and data manipulation within scheduling contexts.
## Input types
### Required
- **`schedule_list`**
    - The 'schedule_list' parameter is a list of numeric values representing a schedule. It is essential for the conversion process, as it undergoes transformation into floats or integers, depending on the context, to ensure compatibility with other modules.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
## Output types
- **`floats`**
    - Comfy dtype: `FLOATS`
    - A list of floats, representing the original schedule list without modification.
    - Python dtype: `List[float]`
- **`float`**
    - Comfy dtype: `FLOAT`
    - A duplicate of the 'floats' list, serving as a redundant output for compatibility purposes.
    - Python dtype: `List[float]`
- **`int`**
    - Comfy dtype: `INT`
    - A list of integers, derived from rounding the values of the original schedule list, to accommodate modules requiring integer inputs.
    - Python dtype: `List[int]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltScheduleConverter:
    """
        Converts a LIST input to FLOATS or FLOAT type
        Makes schedule lists compatible with MTB, IPAdapter, and other modules that use false types.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST",)
            },
        }
    
    RETURN_TYPES = ("FLOATS", "FLOAT", "INT")
    RETURN_NAMES = ("floats", "float", "int")
    FUNCTION = "convert"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def convert(self, schedule_list):
        int_schedule = [int(round(val)) for val in schedule_list]
        return (schedule_list, schedule_list, int_schedule)

```
