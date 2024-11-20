# AppendInstanceDiffusionTracking
## Documentation
- Class name: `AppendInstanceDiffusionTracking`
- Category: `KJNodes/InstanceDiffusion`
- Output node: `False`

The AppendInstanceDiffusionTracking node is designed for integrating and enhancing tracking data within the InstanceDiffusion framework. It merges tracking information from two sources, ensuring that class data is combined without duplication, and concatenates prompts for enriched context. This node facilitates the creation of comprehensive tracking datasets for InstanceDiffusion applications, streamlining the process of data preparation for instance-based diffusion tasks.
## Input types
### Required
- **`tracking_i`**
    - unknown
    - Comfy dtype: `TRACKING`
    - Python dtype: `unknown`
### Optional
- **`prompt_i`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
## Output types
- **`tracking`**
    - Comfy dtype: `TRACKING`
    - The merged tracking data, combining class information and IDs from both input sources without duplication.
    - Python dtype: `dict`
- **`prompt`**
    - Comfy dtype: `STRING`
    - A concatenated string of the two input prompts, providing a unified context for the diffusion task.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AppendInstanceDiffusionTracking:
    
    RETURN_TYPES = ("TRACKING", "STRING",)
    RETURN_NAMES = ("tracking", "prompt",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/InstanceDiffusion"
    DESCRIPTION = """
Appends tracking data to be used with InstanceDiffusion:  
https://github.com/logtd/ComfyUI-InstanceDiffusion  

"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tracking_1": ("TRACKING", {"forceInput": True}),
                "tracking_2": ("TRACKING", {"forceInput": True}),
        },
        "optional": {
            "prompt_1": ("STRING", {"default": "", "forceInput": True}),
            "prompt_2": ("STRING", {"default": "", "forceInput": True}),
        }
    } 

    def append(self, tracking_1, tracking_2, prompt_1="", prompt_2=""):
        tracking_copy = tracking_1.copy()
        # Check for existing class names and class IDs, and raise an error if they exist
        for class_name, class_data in tracking_2.items():
            if class_name not in tracking_copy:
                tracking_copy[class_name] = class_data
            else:
                # If the class name exists, merge the class data from tracking_2 into tracking_copy
                # This will add new class IDs under the same class name without raising an error
                tracking_copy[class_name].update(class_data)
        prompt_string = prompt_1 + "," + prompt_2
        return (tracking_copy, prompt_string)

```
