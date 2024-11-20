---
tags:
- Scheduling
---

# Schedule Random Values
## Documentation
- Class name: `SaltScheduleRandomValues`
- Category: `SALT/AudioViz/Scheduling`
- Output node: `False`

This node is designed to generate a list of random values within a specified range. It allows for the creation of random schedules by specifying a count of values to generate, along with minimum and maximum values. An option to force the generated values to be integers is also provided, enhancing its flexibility for various scheduling and sequencing tasks.
## Input types
### Required
- **`count`**
    - Specifies the number of random values to generate. It determines the length of the resulting schedule list.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`min_value`**
    - The minimum value that can be generated. This sets the lower bound for the random values in the schedule list.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_value`**
    - The maximum value that can be generated. This sets the upper bound for the random values in the schedule list.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`force_integer`**
    - A boolean flag that, when set to true, forces all generated values to be integers, providing control over the numerical type of the schedule list.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`schedule_list`**
    - Comfy dtype: `LIST`
    - The list of randomly generated values, adhering to the specified minimum and maximum values and count. It can consist of either floating-point numbers or integers, depending on the 'force_integer' flag.
    - Python dtype: `List[float] or List[int]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltScheduleRandomValues:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"min": 1}),
                "min_value": ("FLOAT", {}),
                "max_value": ("FLOAT", {}),
            },
            "optional": {
                "force_integer": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list", )
    FUNCTION = "generate_random"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def generate_random(self, count, min_value, max_value, force_integer=False):
        if min_value >= max_value:
            raise ValueError("Schedule min_value must be less than max_value.")
        
        output_type = int if force_integer or (isinstance(min_value, int) and isinstance(max_value, int)) else float
        
        if output_type == int:
            random_schedule = [random.randint(int(min_value), int(max_value)) for _ in range(count)]
        else:
            random_schedule = [random.uniform(min_value, max_value) for _ in range(count)]

        return (random_schedule, )

```
