---
tags:
- Debugging
---

# 🔧 Console Debug
## Documentation
- Class name: `ConsoleDebug+`
- Category: `essentials/utilities`
- Output node: `True`

The ConsoleDebug node is designed for debugging purposes, allowing users to print values to the console with an optional prefix. It facilitates the observation of data flow and values within a node network by providing a simple way to output information to the standard output in a visually distinguishable format.
## Input types
### Required
- **`value`**
    - The primary data or value to be printed. This parameter is essential for the node's operation as it determines what information will be displayed in the console.
    - Comfy dtype: `*`
    - Python dtype: `str | int | float | dict | list | tuple`
### Optional
- **`prefix`**
    - An optional string that precedes the printed value, helping to contextualize or label the output. Its default value is 'Value:'.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConsoleDebug:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (any, {}),
            },
            "optional": {
                "prefix": ("STRING", { "multiline": False, "default": "Value:" })
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "essentials/utilities"
    OUTPUT_NODE = True

    def execute(self, value, prefix):
        print(f"\033[96m{prefix} {value}\033[0m")

        return (None,)

```
