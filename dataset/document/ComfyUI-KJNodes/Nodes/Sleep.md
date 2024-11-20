---
tags:
- Time
---

# Sleep
## Documentation
- Class name: `Sleep`
- Category: `KJNodes/misc`
- Output node: `False`

The Sleep node introduces a delay in the execution flow based on a specified amount of time, allowing for timed pauses in processing sequences.
## Input types
### Required
- **`input`**
    - Acts as a pass-through for any input data, ensuring that the data is returned after the sleep delay without modification.
    - Comfy dtype: `*`
    - Python dtype: `any`
- **`minutes`**
    - Specifies the duration of the delay in whole minutes, contributing to the total sleep time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seconds`**
    - Specifies the duration of the delay in fractional minutes (seconds), allowing for precise control over the total sleep time.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`*`**
    - Comfy dtype: `*`
    - Returns the input data unchanged after the specified delay, maintaining data integrity through the sleep process.
    - Python dtype: `any`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Sleep:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": (any, {}),
                "minutes": ("INT", {"default": 0, "min": 0, "max": 1439}),
                "seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 59.99, "step": 0.01}),
            },
        }
    RETURN_TYPES = (any,)
    FUNCTION = "sleepdelay"
    CATEGORY = "KJNodes/misc"
    DESCRIPTION = """
Delays the execution for the input amount of time.
"""

    def sleepdelay(self, input, minutes, seconds):
        total_seconds = minutes * 60 + seconds
        time.sleep(total_seconds)
        return input,

```
