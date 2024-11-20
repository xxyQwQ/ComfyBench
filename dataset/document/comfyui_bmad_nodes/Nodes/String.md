---
tags:
- Time
---

# String
## Documentation
- Class name: `String`
- Category: `Bmad`
- Output node: `False`

The String node is designed to simply pass through the input string it receives without modification. This node serves as a straightforward utility within a larger processing pipeline, allowing for the direct transfer of string data from one node to another.
## Input types
### Required
- **`inStr`**
    - This parameter represents the input string that will be passed through the node. It is essential for the node's operation as it directly determines the output of the node.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The output of this node is the input string unchanged, facilitating the transfer of string data within a processing pipeline.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"inStr": ("STRING", {"default": ""})}, }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "pass_it"
    CATEGORY = base_category_path

    def pass_it(self, inStr):
        return (inStr,)

```
