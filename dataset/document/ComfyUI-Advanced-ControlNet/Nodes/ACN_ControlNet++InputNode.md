---
tags:
- ControlNet
---

# ControlNet++ Input 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_ControlNet++InputNode`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/ControlNet++`
- Output node: `False`

This node is designed to serve as the entry point for input data into the ControlNet++ architecture, facilitating the preprocessing and preparation of data for further processing within the advanced ControlNet++ framework. It abstracts the complexities of data formatting and ensures compatibility with the ControlNet++ model's requirements.
## Input types
### Required
- **`image`**
    - Specifies the image data to be processed and manipulated by the ControlNet++ model, serving as the foundational input for the architecture.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image or None`
- **`control_type`**
    - Determines the type of control to be applied by the ControlNet++ model, influencing how the input image is processed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`prev_plus_input`**
    - An optional parameter that allows for the chaining of ControlNet++ processing steps by using the output of a previous ControlNet++ node as input.
    - Comfy dtype: `PLUS_INPUT`
    - Python dtype: `PlusInput or None`
- **`autosize`**
    - An optional parameter that automatically adjusts the size of the input image to meet the requirements of the ControlNet++ model, ensuring optimal processing.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `dict or None`
## Output types
- **`plus_input`**
    - Comfy dtype: `PLUS_INPUT`
    - The processed input data ready for further manipulation or analysis within the ControlNet++ framework.
    - Python dtype: `PlusInput`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PlusPlusInputNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "control_type": (PlusPlusType._LIST,),
            },
            "optional": {
                "prev_plus_input": ("PLUS_INPUT",),
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
                #"strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": BIGMAX, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("PLUS_INPUT", )
    FUNCTION = "wrap_images"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/ControlNet++"

    def wrap_images(self, image: Tensor, control_type: str, strength=1.0, prev_plus_input: PlusPlusInputGroup=None):
        if prev_plus_input is None:
            prev_plus_input = PlusPlusInputGroup()
        prev_plus_input = prev_plus_input.clone()

        if math.isclose(strength, 0.0):
            strength = 0.0000001
        pp_input = PlusPlusInput(image, control_type, strength)
        prev_plus_input.add(pp_input)

        return (prev_plus_input,)

```
