---
tags:
- BooleanLogic
- ConditionalSelection
---

# ImpactIfNone
## Documentation
- Class name: `ImpactIfNone`
- Category: `ImpactPack/Logic`
- Output node: `False`

The ImpactIfNone node is designed to evaluate whether a given input is None and conditionally execute logic based on that evaluation. It abstracts the process of null-checking, allowing for streamlined decision-making in data flows where the presence or absence of data dictates subsequent actions.
## Input types
### Required
### Optional
- **`signal`**
    - The 'signal' parameter is an optional input that the node can process alongside 'any_input' to determine the flow of logic based on the presence of data.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_input`**
    - The 'any_input' parameter is evaluated to check if it is None. Its presence or absence influences the node's decision-making process and the output generated.
    - Comfy dtype: `*`
    - Python dtype: `Any`
## Output types
- **`signal_opt`**
    - Comfy dtype: `*`
    - The 'signal_opt' output returns the 'signal' input if 'any_input' is not None, facilitating conditional logic flows.
    - Python dtype: `Any`
- **`bool`**
    - Comfy dtype: `BOOLEAN`
    - The 'bool' output indicates whether 'any_input' was None (False) or not (True), providing a boolean flag for further decision-making.
    - Python dtype: `bool`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImpactIfNone:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {"signal": (any_typ,), "any_input": (any_typ,), }
        }

    RETURN_TYPES = (any_typ, "BOOLEAN")
    RETURN_NAMES = ("signal_opt", "bool")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic"

    def doit(self, signal=None, any_input=None):
        if any_input is None:
            return (signal, False, )
        else:
            return (signal, True, )

```
